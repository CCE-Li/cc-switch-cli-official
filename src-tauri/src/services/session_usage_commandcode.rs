//! Command Code 会话用量导入器
//!
//! 从 `~/.commandcode/projects/<project>/<session>.jsonl` 提取 assistant 消息的
//! token 与费用，写入共享的 `proxy_request_logs` 用量表，让 Command Code 在
//! Usage 里拥有独立的应用栏。
//!
//! ## 行形状
//! ```text
//! {"type":"message","id":"e8a36345","timestamp":"2026-09-29T09:46:06.559Z",
//!  "message":{"role":"assistant","content":[...],"meta":{"source":"model","messageId":"4d639bcc-..."}},
//!  "usage":{"inputTokens":19891,"outputTokens":11,"cacheReadTokens":7424,
//!           "cacheWriteTokens":0,"costUsd":0.001898922},
//!  "model":"deepseek/deepseek-v4-flash"}
//! ```
//!
//! ## 口径
//! `inputTokens` 是**含缓存**的总输入（用 Command Code 官方价目可精确复核
//! `costUsd`），因此落库的是扣掉缓存后的 fresh 输入，并把
//! `input_token_semantics` 标为 FRESH —— 与 `fresh_input_sql` 的归一化口径一致，
//! 无需把本应用加入 `CACHE_INCLUSIVE_APP_TYPES`。
//!
//! 费用直接采信会话里的 `costUsd`：Command Code 有自己的价目表，本地
//! `model_pricing` 里的同名模型价格与之不同，重算只会算错，因此分量成本留 0、
//! 只记总费用。
//!
//! ## 幂等
//! 会话文件变更时整文件重扫，靠 `session_usage_dedup` 持久账本去重。账本是必需的：
//! `rollup_and_prune` 会删除已经汇总的明细行，只靠 `request_id` 查重会在重扫后
//! 重复计数。

use crate::database::{lock_conn, Database};
use crate::error::AppError;
use crate::services::session_usage::{metadata_modified_nanos, SessionSyncResult};
use crate::services::sql_helpers::INPUT_TOKEN_SEMANTICS_FRESH;
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

const APP_TYPE: &str = "commandcode";
const DATA_SOURCE: &str = "commandcode_session";
const PROVIDER_PLACEHOLDER: &str = "_commandcode_session";
const UNKNOWN_MODEL: &str = "unknown";
const CHECKPOINTS_SUFFIX: &str = ".checkpoints.jsonl";
/// 单文件安全上限：Command Code 的会话日志是追加写的 JSONL，超过这个体量说明
/// 路径并非会话日志（或已损坏），宁可报错也不做无界读取。
const MAX_SESSION_BYTES: u64 = 64 * 1024 * 1024;
const MAX_USAGE_LABEL_BYTES: usize = 512;
const MIN_SQLITE_UNIX_SECONDS: i64 = -62_167_219_200;
const MAX_SQLITE_UNIX_SECONDS: i64 = 253_402_300_799;

const REQUEST_DEDUP_SQL: &str = "SELECT EXISTS(
         SELECT 1 FROM session_usage_dedup
         WHERE data_source = ?1 AND request_id = ?2
     )";
const SEMANTIC_DEDUP_SQL: &str = "SELECT EXISTS(
         SELECT 1 FROM session_usage_dedup
         WHERE data_source = ?1 AND semantic_id = ?2
     )";
const LEGACY_SEMANTIC_DEDUP_SQL: &str = "SELECT EXISTS(
         SELECT 1 FROM session_usage_dedup
         WHERE data_source = ?1 AND semantic_id = ?2 AND has_entry_id = 0
     )";

/// 一条 assistant 消息对应的用量记录。
#[derive(Debug)]
struct CommandCodeUsageRecord {
    request_id: String,
    semantic_id: String,
    has_entry_id: bool,
    model: String,
    input_tokens: u32,
    output_tokens: u32,
    cache_read_tokens: u32,
    cache_creation_tokens: u32,
    total_cost: Decimal,
    created_at: i64,
    session_id: String,
}

#[derive(Debug)]
struct ParsedSessionFile {
    records: Vec<CommandCodeUsageRecord>,
    /// 文件末尾存在未写完的行：本次不推进游标，下一轮重扫。
    incomplete_tail: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CommandCodeSyncState {
    modified_nanos: i64,
    byte_size: u64,
}

/// 窄结构体：只反序列化用量追踪需要的字段，避免为整行（含大段 tool 结果）
/// 构建完整的 `serde_json::Value`。
#[derive(Debug, Deserialize)]
struct NarrowCommandCodeLine {
    #[serde(rename = "type")]
    kind: Option<String>,
    id: Option<String>,
    timestamp: Option<String>,
    message: Option<NarrowCommandCodeMessage>,
    usage: Option<NarrowCommandCodeUsage>,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NarrowCommandCodeMessage {
    role: Option<String>,
    meta: Option<NarrowCommandCodeMessageMeta>,
}

#[derive(Debug, Deserialize)]
struct NarrowCommandCodeMessageMeta {
    #[serde(rename = "messageId")]
    message_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct NarrowCommandCodeUsage {
    #[serde(rename = "inputTokens", default)]
    input_tokens: u64,
    #[serde(rename = "outputTokens", default)]
    output_tokens: u64,
    #[serde(rename = "cacheReadTokens", default)]
    cache_read_tokens: u64,
    #[serde(rename = "cacheWriteTokens", default)]
    cache_write_tokens: u64,
    #[serde(rename = "costUsd")]
    cost_usd: Option<Value>,
}

/// 导入 Command Code 的全部会话用量。
pub fn sync_commandcode_usage(db: &Database) -> Result<SessionSyncResult, AppError> {
    let root = commandcode_projects_dir();
    if !root.exists() {
        return Ok(SessionSyncResult::default());
    }

    let files = collect_session_files(&root);
    let states = load_sync_states(db, &root)?;
    let mut result = SessionSyncResult {
        files_scanned: files.len().min(u32::MAX as usize) as u32,
        ..Default::default()
    };

    for (path, modified_nanos, byte_size) in files {
        let previous = states.get(&path.to_string_lossy().to_string()).copied();
        match sync_single_session_file(db, &path, modified_nanos, byte_size, previous) {
            Ok(file_result) => result.merge(file_result),
            Err(error) => {
                let message = format!("{}: {error}", path.display());
                log::warn!("[COMMANDCODE-SYNC] 会话文件解析失败: {message}");
                result.errors.push(message);
            }
        }
    }

    if result.imported > 0 {
        log::info!(
            "[COMMANDCODE-SYNC] 同步完成: 导入 {} 条, 跳过 {} 条, 扫描 {} 个文件",
            result.imported,
            result.skipped,
            result.files_scanned
        );
    }
    Ok(result)
}

/// Command Code 配置目录（`~/.commandcode`）。
fn commandcode_config_dir() -> PathBuf {
    crate::config::get_home_dir().join(".commandcode")
}

fn commandcode_projects_dir() -> PathBuf {
    commandcode_config_dir().join("projects")
}

/// 收集 `projects/<项目>/<会话>.jsonl` 并按 mtime 降序返回 `(路径, mtime, 大小)`。
///
/// `<会话>.checkpoints.jsonl` 是另一条日志流，不携带用量，必须排除。
fn collect_session_files(root: &Path) -> Vec<(PathBuf, i64, u64)> {
    let mut files = Vec::new();
    let Ok(project_dirs) = fs::read_dir(root) else {
        return files;
    };

    for project_dir in project_dirs.flatten() {
        let project_path = project_dir.path();
        if !project_path.is_dir() {
            continue;
        }
        let Ok(entries) = fs::read_dir(&project_path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|name| name.to_str());
            if !name.is_some_and(|name| name.ends_with(".jsonl"))
                || name.is_some_and(|name| name.ends_with(CHECKPOINTS_SUFFIX))
            {
                continue;
            }
            // symlink_metadata beyond metadata: 会话日志必须是普通文件，避免
            // 顺着符号链接把任意路径读进用量表。
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            if !metadata.file_type().is_file() {
                continue;
            }
            files.push((path, metadata_modified_nanos(&metadata), metadata.len()));
        }
    }

    files.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.1));
    files
}

fn load_sync_states(
    db: &Database,
    projects_root: &Path,
) -> Result<HashMap<String, CommandCodeSyncState>, AppError> {
    let conn = lock_conn!(db.conn);
    let mut stmt = conn
        .prepare("SELECT file_path, last_modified, last_byte_offset FROM session_log_sync")
        .map_err(|error| AppError::Database(format!("预取 Command Code 同步游标失败: {error}")))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<i64>>(2)?,
            ))
        })
        .map_err(|error| AppError::Database(format!("预取 Command Code 同步游标失败: {error}")))?;

    // 只挑本模块自己写下的游标：Claude 导入器也会写 last_byte_offset，
    // 所以这里除了该列必须存在，还要按会话根目录前缀过滤。
    let prefix = projects_root.to_string_lossy().to_string();
    let mut states = HashMap::new();
    for row in rows {
        let (file_path, modified_nanos, byte_offset) = row
            .map_err(|error| AppError::Database(format!("读取 Command Code 游标失败: {error}")))?;
        if !file_path.starts_with(&prefix) {
            continue;
        }
        if let Some(byte_offset) = byte_offset.filter(|offset| *offset >= 0) {
            states.insert(
                file_path,
                CommandCodeSyncState {
                    modified_nanos,
                    byte_size: byte_offset as u64,
                },
            );
        }
    }
    Ok(states)
}

fn sync_single_session_file(
    db: &Database,
    file_path: &Path,
    file_modified_nanos: i64,
    byte_size: u64,
    previous: Option<CommandCodeSyncState>,
) -> Result<SessionSyncResult, AppError> {
    if previous.is_some_and(|state| {
        state.modified_nanos == file_modified_nanos && state.byte_size == byte_size
    }) {
        return Ok(SessionSyncResult::default());
    }
    if byte_size > MAX_SESSION_BYTES {
        return Err(AppError::Config(format!(
            "Command Code 会话文件超过 {MAX_SESSION_BYTES} 字节安全上限"
        )));
    }

    let session_id = file_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default()
        .to_string();
    let file = File::open(file_path)
        .map_err(|error| AppError::Config(format!("无法打开 Command Code 会话文件: {error}")))?;
    let parsed = parse_session_file(file, &session_id, file_modified_nanos)?;

    let conn = lock_conn!(db.conn);
    let tx = conn.unchecked_transaction().map_err(|error| {
        AppError::Database(format!("启动 Command Code 用量导入事务失败: {error}"))
    })?;

    let mut result = SessionSyncResult::default();
    for record in &parsed.records {
        if insert_usage_record(&tx, record)? {
            result.imported = result.imported.saturating_add(1);
        } else {
            result.skipped = result.skipped.saturating_add(1);
        }
    }

    if parsed.incomplete_tail {
        // 末行没写完：不推进游标，下一轮重扫（账本保证不会重复计数）。
        result.deferred_files = 1;
    } else {
        update_sync_state(
            &tx,
            &file_path.to_string_lossy(),
            file_modified_nanos,
            byte_size,
        )?;
    }
    tx.commit()
        .map_err(|error| AppError::Database(format!("提交 Command Code 用量导入失败: {error}")))?;
    Ok(result)
}

/// 逐行解析会话文件。坏行（非 JSON / 结构不符）跳过，末行缺换行视为未写完。
fn parse_session_file(
    file: File,
    session_id: &str,
    file_modified_nanos: i64,
) -> Result<ParsedSessionFile, AppError> {
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    let mut records = Vec::new();
    let mut incomplete_tail = false;
    let file_timestamp = file_modified_nanos / 1_000_000_000;

    loop {
        buffer.clear();
        let read = reader.read_until(b'\n', &mut buffer).map_err(|error| {
            AppError::Config(format!("无法读取 Command Code 会话文件: {error}"))
        })?;
        if read == 0 {
            break;
        }
        let complete = buffer.last() == Some(&b'\n');
        let Ok(line) = std::str::from_utf8(&buffer) else {
            if !complete {
                incomplete_tail = true;
                break;
            }
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // 预过滤：只有 assistant 行可能带 usage，避免为大段 tool 结果建结构。
        let parsed = if line.contains("assistant") {
            serde_json::from_str::<NarrowCommandCodeLine>(line).ok()
        } else {
            None
        };
        match parsed {
            Some(parsed) => {
                if let Some(record) = parse_usage_record(&parsed, session_id, file_timestamp) {
                    records.push(record);
                }
            }
            None if !complete => {
                incomplete_tail = true;
                break;
            }
            None => {}
        }
    }

    Ok(ParsedSessionFile {
        records,
        incomplete_tail,
    })
}

fn parse_usage_record(
    parsed: &NarrowCommandCodeLine,
    session_id: &str,
    file_timestamp: i64,
) -> Option<CommandCodeUsageRecord> {
    if parsed.kind.as_deref() != Some("message") {
        return None;
    }
    let message = parsed.message.as_ref()?;
    if message.role.as_deref() != Some("assistant") {
        return None;
    }
    let usage = parsed.usage.as_ref()?;

    let cache_read_tokens = to_u32(usage.cache_read_tokens);
    let cache_creation_tokens = to_u32(usage.cache_write_tokens);
    // inputTokens 是含缓存的总输入，落库前折算成 fresh 输入。这里把 cacheWrite
    // 也一并扣除：它与 fresh_input_sql 的 TOTAL 口径一致（即总输入 = fresh +
    // cacheRead + cacheWrite）。实测 1300+ 条真实记录 cacheWriteTokens 恒为 0，
    // 所以两种解释目前等价。
    let fresh_input = usage
        .input_tokens
        .saturating_sub(usage.cache_read_tokens)
        .saturating_sub(usage.cache_write_tokens);
    let input_tokens = to_u32(fresh_input);
    let output_tokens = to_u32(usage.output_tokens);
    let total_cost = parse_reported_cost(usage.cost_usd.as_ref());

    if input_tokens == 0
        && output_tokens == 0
        && cache_read_tokens == 0
        && cache_creation_tokens == 0
        && total_cost == Decimal::ZERO
    {
        return None;
    }

    let model = bounded_label(parsed.model.as_deref(), UNKNOWN_MODEL);
    let created_at = parsed
        .timestamp
        .as_deref()
        .and_then(parse_timestamp_seconds)
        .unwrap_or(file_timestamp)
        .clamp(MIN_SQLITE_UNIX_SECONDS, MAX_SQLITE_UNIX_SECONDS);

    let entry_id = nonempty(
        message
            .meta
            .as_ref()
            .and_then(|meta| meta.message_id.as_deref()),
    )
    .or_else(|| nonempty(parsed.id.as_deref()));
    let has_entry_id = entry_id.is_some();
    let semantic_id = semantic_identity(
        session_id,
        entry_id,
        &model,
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        created_at,
    );
    let request_id = match entry_id {
        // 会话 id 必须进身份：行内 `id` 只有 32 位，跨会话撞车时会被当成重复
        // 而静默丢掉用量，所以这里和 opencode/gemini 导入器一样带上会话 id。
        Some(entry_id) => format!("{DATA_SOURCE}:{session_id}:{}", truncate_label(entry_id)),
        None => semantic_id.clone(),
    };

    Some(CommandCodeUsageRecord {
        request_id,
        semantic_id,
        has_entry_id,
        model,
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        total_cost,
        created_at,
        session_id: session_id.to_string(),
    })
}

// 语义身份需要覆盖全部用量字段，逐项传参比包一个只用一次的结构体更直观。
#[allow(clippy::too_many_arguments)]
fn semantic_identity(
    session_id: &str,
    entry_id: Option<&str>,
    model: &str,
    input_tokens: u32,
    output_tokens: u32,
    cache_read_tokens: u32,
    cache_creation_tokens: u32,
    created_at: i64,
) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, b"commandcode-session-semantic-v1");
    hash_field(&mut hasher, session_id.as_bytes());
    hash_field(&mut hasher, entry_id.unwrap_or_default().as_bytes());
    hash_field(&mut hasher, model.as_bytes());
    hash_field(&mut hasher, &input_tokens.to_be_bytes());
    hash_field(&mut hasher, &output_tokens.to_be_bytes());
    hash_field(&mut hasher, &cache_read_tokens.to_be_bytes());
    hash_field(&mut hasher, &cache_creation_tokens.to_be_bytes());
    hash_field(&mut hasher, &created_at.to_be_bytes());
    format!("{DATA_SOURCE}_semantic:{:x}", hasher.finalize())
}

fn hash_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn insert_usage_record(
    conn: &rusqlite::Connection,
    record: &CommandCodeUsageRecord,
) -> Result<bool, AppError> {
    let request_seen: bool = conn
        .query_row(
            REQUEST_DEDUP_SQL,
            rusqlite::params![DATA_SOURCE, record.request_id],
            |row| row.get(0),
        )
        .map_err(|error| {
            AppError::Database(format!("查询 Command Code 用量去重账本失败: {error}"))
        })?;
    let already_seen = request_seen
        || conn
            .query_row(
                if record.has_entry_id {
                    LEGACY_SEMANTIC_DEDUP_SQL
                } else {
                    SEMANTIC_DEDUP_SQL
                },
                rusqlite::params![DATA_SOURCE, record.semantic_id],
                |row| row.get(0),
            )
            .map_err(|error| {
                AppError::Database(format!("查询 Command Code 用量去重账本失败: {error}"))
            })?;
    if already_seen {
        return Ok(false);
    }

    conn.execute(
        "INSERT OR IGNORE INTO session_usage_dedup
         (data_source, request_id, semantic_id, has_entry_id)
         VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![
            DATA_SOURCE,
            record.request_id,
            record.semantic_id,
            i64::from(record.has_entry_id),
        ],
    )
    .map_err(|error| AppError::Database(format!("写入 Command Code 用量去重账本失败: {error}")))?;

    conn.execute(
        "INSERT OR IGNORE INTO proxy_request_logs (
            request_id, provider_id, app_type, model, request_model, pricing_model,
            input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
            input_token_semantics,
            input_cost_usd, output_cost_usd, cache_read_cost_usd,
            cache_creation_cost_usd, total_cost_usd,
            latency_ms, first_token_ms, status_code, error_message, session_id,
            provider_type, is_streaming, cost_multiplier, created_at, data_source
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
            ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26
        )",
        rusqlite::params![
            record.request_id,
            PROVIDER_PLACEHOLDER,
            APP_TYPE,
            record.model,
            record.model,
            record.model,
            record.input_tokens,
            record.output_tokens,
            record.cache_read_tokens,
            record.cache_creation_tokens,
            INPUT_TOKEN_SEMANTICS_FRESH,
            // 分量成本留 0：Command Code 的价目不在本地定价表里，只有总费用可信。
            "0",
            "0",
            "0",
            "0",
            record.total_cost.to_string(),
            0i64,
            Option::<i64>::None,
            200i64,
            Option::<String>::None,
            record.session_id,
            Some(DATA_SOURCE),
            1i64,
            "1.0",
            record.created_at,
            DATA_SOURCE,
        ],
    )
    .map(|changed| changed > 0)
    .map_err(|error| AppError::Database(format!("插入 Command Code 会话用量失败: {error}")))
}

/// 写入本模块专用的游标：`last_byte_offset` 存已扫描的快照字节数，
/// `last_line_offset` 保持 0（本模块按整文件重扫，不用行游标）。
fn update_sync_state(
    conn: &rusqlite::Connection,
    file_path: &str,
    modified_nanos: i64,
    byte_size: u64,
) -> Result<(), AppError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    let byte_offset = i64::try_from(byte_size)
        .map_err(|_| AppError::Config("Command Code 会话日志过大，无法保存字节游标".to_string()))?;

    conn.execute(
        "INSERT INTO session_log_sync
             (file_path, last_modified, last_line_offset, last_synced_at, last_byte_offset,
              last_tail_fingerprint)
         VALUES (?1, ?2, 0, ?3, ?4, NULL)
         ON CONFLICT(file_path) DO UPDATE SET
            last_modified = excluded.last_modified,
            last_line_offset = 0,
            last_synced_at = excluded.last_synced_at,
            last_byte_offset = excluded.last_byte_offset,
            last_tail_fingerprint = NULL",
        rusqlite::params![file_path, modified_nanos, now, byte_offset],
    )
    .map_err(|error| AppError::Database(format!("更新 Command Code 同步游标失败: {error}")))?;
    Ok(())
}

fn to_u32(value: u64) -> u32 {
    value.min(u32::MAX as u64) as u32
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn bounded_label(value: Option<&str>, fallback: &str) -> String {
    truncate_label(nonempty(value).unwrap_or(fallback)).to_string()
}

fn truncate_label(value: &str) -> &str {
    if value.len() <= MAX_USAGE_LABEL_BYTES {
        return value;
    }
    let mut end = MAX_USAGE_LABEL_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

/// `costUsd` 可能是 JSON 数字或字符串；缺失/不可解析按 0 处理。
fn parse_reported_cost(value: Option<&Value>) -> Decimal {
    let Some(value) = value else {
        return Decimal::ZERO;
    };
    let raw = match value {
        Value::Number(number) => number.to_string(),
        Value::String(text) => text.clone(),
        _ => return Decimal::ZERO,
    };
    Decimal::from_str(&raw)
        .or_else(|_| Decimal::from_scientific(&raw))
        .map(|cost| cost.max(Decimal::ZERO))
        .unwrap_or(Decimal::ZERO)
}

fn parse_timestamp_seconds(value: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|timestamp| timestamp.timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const USAGE_LINE: &str = r#"{"type":"message","id":"e8a36345","parentId":null,"timestamp":"2026-09-29T09:46:06.559Z","message":{"role":"assistant","content":[{"type":"text","text":"ok"}],"meta":{"source":"model","messageId":"4d639bcc-fdc9-4965-bede-ff4d6723063c"}},"usage":{"inputTokens":19891,"outputTokens":11,"cacheReadTokens":7424,"cacheWriteTokens":0,"costUsd":0.001898922},"model":"deepseek/deepseek-v4-flash"}"#;
    const USER_LINE: &str = r#"{"type":"message","id":"7d8268bb","parentId":null,"timestamp":"2026-09-29T09:46:06.559Z","message":{"role":"user","content":[{"type":"text","text":"hi"}],"meta":{"source":"user","messageId":"7194ad24-812b-477f-a5b5-8592914cab63"}}}"#;

    fn write_session(dir: &Path, name: &str, lines: &[&str]) -> PathBuf {
        fs::create_dir_all(dir).expect("create project dir");
        let path = dir.join(name);
        let mut file = File::create(&path).expect("create session file");
        for line in lines {
            writeln!(file, "{line}").expect("write session line");
        }
        path
    }

    fn projects_dir(home: &Path) -> PathBuf {
        home.join(".commandcode")
            .join("projects")
            .join("c-users-test")
    }

    /// 真实形状的一行：token 折算成 fresh 输入、总费用采信 costUsd、
    /// 落进 commandcode 应用与 commandcode_session 数据源。
    #[test]
    fn imports_fresh_input_and_reported_cost() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        write_session(
            &projects_dir(home.path()),
            "session-a.jsonl",
            &[
                r#"{"type":"session","version":3,"id":"session-a","timestamp":"2026-09-29T09:16:57.098Z","cwd":"C:\\work"}"#,
                USER_LINE,
                USAGE_LINE,
            ],
        );
        let db = Database::memory()?;

        let result = sync_commandcode_usage(&db)?;
        assert_eq!(result.imported, 1);
        assert_eq!(result.files_scanned, 1);
        assert!(result.errors.is_empty());

        let conn = lock_conn!(db.conn);
        let row: (String, String, String, i64, i64, i64, i64, i64, String) = conn.query_row(
            "SELECT app_type, data_source, provider_id, input_tokens, output_tokens,
                    cache_read_tokens, cache_creation_tokens, input_token_semantics,
                    total_cost_usd
             FROM proxy_request_logs",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                ))
            },
        )?;
        assert_eq!(row.0, APP_TYPE);
        assert_eq!(row.1, DATA_SOURCE);
        assert_eq!(row.2, PROVIDER_PLACEHOLDER);
        // 19891 - 7424 = 12467 fresh input
        assert_eq!(row.3, 12467);
        assert_eq!(row.4, 11);
        assert_eq!(row.5, 7424);
        assert_eq!(row.6, 0);
        assert_eq!(row.7, INPUT_TOKEN_SEMANTICS_FRESH);
        assert_eq!(
            Decimal::from_str(&row.8).expect("reported total"),
            Decimal::from_str("0.001898922").expect("expected total")
        );
        Ok(())
    }

    /// 重扫幂等：文件未变直接跳过，账本也拦得住重复。
    #[test]
    fn resync_is_idempotent() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let path = write_session(
            &projects_dir(home.path()),
            "session-b.jsonl",
            &[USER_LINE, USAGE_LINE],
        );
        let db = Database::memory()?;

        assert_eq!(sync_commandcode_usage(&db)?.imported, 1);
        // 未变更：游标命中，连文件都不读。
        assert_eq!(sync_commandcode_usage(&db)?.imported, 0);

        // 变更（追加一条新消息）后重扫，历史行靠账本去重。
        let second = USAGE_LINE
            .replace("e8a36345", "e8a36346")
            .replace(
                "4d639bcc-fdc9-4965-bede-ff4d6723063c",
                "4d639bcc-fdc9-4965-bede-ff4d6723063d",
            )
            .replace("19891", "20000");
        {
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("append session");
            writeln!(file, "{second}").expect("append message");
        }

        let appended = sync_commandcode_usage(&db)?;
        assert_eq!(appended.imported, 1);
        assert_eq!(appended.skipped, 1);

        let count: i64 = lock_conn!(db.conn).query_row(
            "SELECT COUNT(*) FROM proxy_request_logs WHERE app_type = ?1",
            rusqlite::params![APP_TYPE],
            |row| row.get(0),
        )?;
        assert_eq!(count, 2);
        Ok(())
    }

    /// `<会话>.checkpoints.jsonl` 不携带用量，必须被排除。
    #[test]
    fn excludes_checkpoint_files() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let project = projects_dir(home.path());
        write_session(&project, "session-c.jsonl", &[USAGE_LINE]);
        write_session(&project, "session-c.checkpoints.jsonl", &[USAGE_LINE]);
        let db = Database::memory()?;

        let result = sync_commandcode_usage(&db)?;
        assert_eq!(result.files_scanned, 1);
        assert_eq!(result.imported, 1);
        Ok(())
    }

    /// 非 assistant / 无 usage / 全零的行不入库。
    #[test]
    fn skips_non_assistant_and_empty_usage() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let zero = USAGE_LINE.replace(
            r#""inputTokens":19891,"outputTokens":11,"cacheReadTokens":7424,"cacheWriteTokens":0,"costUsd":0.001898922"#,
            r#""inputTokens":0,"outputTokens":0,"cacheReadTokens":0,"cacheWriteTokens":0,"costUsd":0"#,
        );
        write_session(
            &projects_dir(home.path()),
            "session-d.jsonl",
            &[
                USER_LINE,
                &zero,
                r#"{"type":"summary","nonsense":true}"#,
                "{not json",
            ],
        );
        let db = Database::memory()?;

        let result = sync_commandcode_usage(&db)?;
        assert_eq!(result.imported, 0);
        assert!(result.errors.is_empty());
        Ok(())
    }

    /// 费用写成字符串也要能解析，且模型名过长时按 UTF-8 边界截断。
    #[test]
    fn parses_string_cost_and_bounds_labels() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let oversized = "界".repeat(MAX_USAGE_LABEL_BYTES);
        let line = USAGE_LINE
            .replace("\"costUsd\":0.001898922", "\"costUsd\":\"0.5\"")
            .replace("deepseek/deepseek-v4-flash", &oversized);
        write_session(&projects_dir(home.path()), "session-e.jsonl", &[&line]);
        let db = Database::memory()?;

        assert_eq!(sync_commandcode_usage(&db)?.imported, 1);
        let conn = lock_conn!(db.conn);
        let (cost, model): (String, String) = conn.query_row(
            "SELECT total_cost_usd, model FROM proxy_request_logs",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(
            Decimal::from_str(&cost).expect("string cost"),
            Decimal::from_str("0.5").expect("expected cost")
        );
        assert!(model.len() <= MAX_USAGE_LABEL_BYTES);
        assert!(std::str::from_utf8(model.as_bytes()).is_ok());
        Ok(())
    }

    /// 末行没写完时不推进游标，补全后能把那条补进来。
    #[test]
    fn incomplete_tail_defers_cursor_until_the_line_completes() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let project = projects_dir(home.path());
        fs::create_dir_all(&project).expect("create project dir");
        let path = project.join("session-f.jsonl");
        let split = USAGE_LINE.len() / 2;
        {
            let mut file = File::create(&path).expect("create partial session");
            file.write_all(&USAGE_LINE.as_bytes()[..split])
                .expect("write partial line");
        }
        let db = Database::memory()?;

        let deferred = sync_commandcode_usage(&db)?;
        assert_eq!(deferred.imported, 0);
        assert_eq!(deferred.deferred_files, 1);

        {
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("append session");
            file.write_all(&USAGE_LINE.as_bytes()[split..])
                .expect("finish line");
            writeln!(file).expect("finish newline");
        }

        let completed = sync_commandcode_usage(&db)?;
        assert_eq!(completed.imported, 1);
        Ok(())
    }

    /// 没有稳定的 messageId 时用内容语义身份去重，重写不会重复计数。
    #[test]
    fn rewrite_without_ids_does_not_double_count() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let project = projects_dir(home.path());
        let no_ids = USAGE_LINE
            .replace(r#""id":"e8a36345","parentId":null,"#, "")
            .replace(
                r#""meta":{"source":"model","messageId":"4d639bcc-fdc9-4965-bede-ff4d6723063c"}"#,
                r#""meta":{"source":"model"}"#,
            );
        let path = write_session(&project, "session-g.jsonl", &[&no_ids]);
        let db = Database::memory()?;

        assert_eq!(sync_commandcode_usage(&db)?.imported, 1);
        // 同内容重写（mtime 变化触发重扫）→ 语义身份命中。
        let rewritten = write_session(&project, "session-g.jsonl", &[&no_ids]);
        assert_eq!(rewritten, path);
        assert_eq!(sync_commandcode_usage(&db)?.imported, 0);

        let count: i64 = lock_conn!(db.conn).query_row(
            "SELECT COUNT(*) FROM proxy_request_logs WHERE app_type = ?1",
            rusqlite::params![APP_TYPE],
            |row| row.get(0),
        )?;
        assert_eq!(count, 1);
        Ok(())
    }

    /// 已汇总并清理明细后，重扫依然不会把历史行当新行导入。
    #[test]
    fn rollup_then_prune_does_not_reimport_history() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let project = projects_dir(home.path());
        // 时间戳必须早于 rollup 的保留窗口，否则明细行不会被汇总清理。
        let old_line = USAGE_LINE.replace("2026-09-29T09:46:06.559Z", "2020-01-01T00:00:00.000Z");
        let path = write_session(&project, "session-h.jsonl", &[&old_line]);
        let db = Database::memory()?;

        assert_eq!(sync_commandcode_usage(&db)?.imported, 1);
        assert_eq!(db.rollup_and_prune(30)?, 1);
        {
            let conn = lock_conn!(db.conn);
            let details: i64 = conn.query_row(
                "SELECT COUNT(*) FROM proxy_request_logs WHERE app_type = ?1",
                rusqlite::params![APP_TYPE],
                |row| row.get(0),
            )?;
            assert_eq!(details, 0, "汇总后明细行应被清理");
        }

        let second = old_line.replace("e8a36345", "e8a36347").replace(
            "4d639bcc-fdc9-4965-bede-ff4d6723063c",
            "4d639bcc-fdc9-4965-bede-ff4d6723063e",
        );
        {
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("append session");
            writeln!(file, "{second}").expect("append message");
        }
        let rescan = sync_commandcode_usage(&db)?;
        assert_eq!(rescan.imported, 1);
        // 账本挡住了已经汇总并清理掉的历史行。
        assert_eq!(rescan.skipped, 1);

        let conn = lock_conn!(db.conn);
        let rolled_up: i64 = conn.query_row(
            "SELECT COALESCE(SUM(request_count), 0) FROM usage_daily_rollups WHERE app_type = ?1",
            rusqlite::params![APP_TYPE],
            |row| row.get(0),
        )?;
        let details: i64 = conn.query_row(
            "SELECT COUNT(*) FROM proxy_request_logs WHERE app_type = ?1",
            rusqlite::params![APP_TYPE],
            |row| row.get(0),
        )?;
        // 两次真实请求 = 汇总 1 条 + 明细 1 条；历史行没有被重复计入。
        assert_eq!(rolled_up, 1);
        assert_eq!(details, 1);
        Ok(())
    }

    /// cacheWrite 非 0 时：总输入 = fresh + cacheRead + cacheWrite。
    #[test]
    fn fresh_input_excludes_cache_write_when_present() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let line = USAGE_LINE.replace(
            r#""inputTokens":19891,"outputTokens":11,"cacheReadTokens":7424,"cacheWriteTokens":0,"costUsd":0.001898922"#,
            r#""inputTokens":100,"outputTokens":7,"cacheReadTokens":10,"cacheWriteTokens":5,"costUsd":0.5"#,
        );
        write_session(&projects_dir(home.path()), "session-i.jsonl", &[&line]);
        let db = Database::memory()?;

        assert_eq!(sync_commandcode_usage(&db)?.imported, 1);
        let conn = lock_conn!(db.conn);
        let (fresh, out, read, write): (i64, i64, i64, i64) = conn.query_row(
            "SELECT input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
             FROM proxy_request_logs WHERE app_type = ?1",
            rusqlite::params![APP_TYPE],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
        assert_eq!((fresh, out, read, write), (85, 7, 10, 5));
        Ok(())
    }

    /// 32 位的行内 id 跨会话会撞车：带上会话 id 后两条都必须入库。
    #[test]
    fn identical_line_ids_in_different_sessions_do_not_collide() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let project = projects_dir(home.path());
        // 去掉 meta.messageId，只留下行内 8 位 id（两个文件里故意完全相同）。
        let no_message_id =
            USAGE_LINE.replace(r#","messageId":"4d639bcc-fdc9-4965-bede-ff4d6723063c""#, "");
        assert!(
            !no_message_id.contains("messageId"),
            "fixture must not carry a messageId"
        );
        write_session(&project, "session-j1.jsonl", &[&no_message_id]);
        write_session(&project, "session-j2.jsonl", &[&no_message_id]);
        let db = Database::memory()?;

        let result = sync_commandcode_usage(&db)?;
        assert_eq!(result.imported, 2);
        assert_eq!(result.skipped, 0);
        Ok(())
    }

    /// 超过安全上限的文件要报错跳过，而不是无界读取。
    #[test]
    fn oversized_file_is_reported_instead_of_silently_skipped() -> Result<(), AppError> {
        let home = tempfile::tempdir().expect("temp home");
        let _env = crate::test_support::TestEnvGuard::isolated(home.path());
        let project = projects_dir(home.path());
        fs::create_dir_all(&project).expect("create project dir");
        File::create(project.join("session-k.jsonl"))
            .expect("create sparse session")
            .set_len(MAX_SESSION_BYTES + 1)
            .expect("size sparse session");
        let db = Database::memory()?;

        let result = sync_commandcode_usage(&db)?;
        assert_eq!(result.files_scanned, 1);
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].contains("安全上限"));
        Ok(())
    }

    /// 去重查询必须命中完整身份索引。
    #[test]
    fn dedup_lookups_use_complete_identity_indexes() -> Result<(), AppError> {
        let db = Database::memory()?;
        let conn = lock_conn!(db.conn);
        for (sql, expected) in [
            (REQUEST_DEDUP_SQL, "(data_source=? AND request_id=?)"),
            (SEMANTIC_DEDUP_SQL, "(data_source=? AND semantic_id=?)"),
            (
                LEGACY_SEMANTIC_DEDUP_SQL,
                "(data_source=? AND semantic_id=? AND has_entry_id=?)",
            ),
        ] {
            let mut statement = conn.prepare(&format!("EXPLAIN QUERY PLAN {sql}"))?;
            let plan = statement
                .query_map(rusqlite::params![DATA_SOURCE, "identity"], |row| {
                    row.get::<_, String>(3)
                })?
                .collect::<Result<Vec<_>, _>>()?;
            assert!(
                plan.iter().any(|step| step.contains(expected)),
                "lookup does not constrain the complete identity {expected}: {plan:?}"
            );
        }
        Ok(())
    }
}
