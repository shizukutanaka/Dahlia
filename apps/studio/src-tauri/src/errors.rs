//! Turn internal error strings into user-facing Japanese (W58/W59).
//!
//! Core and the Python workers already produce Japanese for the
//! cases they own; what still arrives here in English is mostly
//! infrastructure detail (Python exceptions, OS errors, library
//! messages). Known patterns become a plain explanation plus what
//! the user can do next; anything else is wrapped so the screen
//! never shows a bare English technical string, while the detail
//! stays available in parentheses for problem reports.
//!
//! Pure std on purpose: this file is unit-tested standalone.

/// True when the text already contains Japanese (hiragana, katakana
/// or CJK ideographs) and can be shown to the user as-is.
fn has_japanese(s: &str) -> bool {
    s.chars()
        .any(|c| ('\u{3040}'..='\u{30ff}').contains(&c) || ('\u{4e00}'..='\u{9fff}').contains(&c))
}

pub fn friendly(raw: &str) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        return "不明な問題が発生しました。もう一度お試しください".to_string();
    }
    if has_japanese(raw) {
        return raw.to_string();
    }
    let lower = raw.to_lowercase();
    if lower.contains("no such file") || lower.contains("not found") && lower.contains(".onnx") {
        return format!(
            "必要なファイルが見つかりません。アプリの配置が正しいか確認してください（詳細: {raw}）"
        );
    }
    if lower.contains("onnxruntime") || lower.contains("modulenotfounderror") {
        return format!(
            "AI処理の実行環境が不足しています。配布パッケージ一式が揃っているか確認してください（詳細: {raw}）"
        );
    }
    if lower.contains("permission denied") || lower.contains("access is denied") {
        return format!(
            "ファイルへのアクセスが拒否されました。保存先の権限やセキュリティソフトを確認してください（詳細: {raw}）"
        );
    }
    if lower.contains("no space") || lower.contains("disk full") {
        return format!(
            "ディスクの空き容量が不足しています。空きを確保してからもう一度お試しください（詳細: {raw}）"
        );
    }
    if lower.contains("database") || lower.contains("sqlite") {
        return format!(
            "データベースの処理で問題が発生しました。アプリを再起動しても直らない場合は、ログを添えて相談してください（詳細: {raw}）"
        );
    }
    if lower.contains("image") || lower.contains("cannot identify") {
        return format!(
            "画像を読み取れませんでした。対応形式（JPEG / PNG 等）のファイルか確認してください（詳細: {raw}）"
        );
    }
    format!(
        "処理中に問題が発生しました。もう一度お試しても直らない場合は、ログを添えて相談してください（詳細: {raw}）"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn japanese_passes_through() {
        assert_eq!(
            friendly("特徴量がまだ取得されていません"),
            "特徴量がまだ取得されていません"
        );
    }

    #[test]
    fn english_is_wrapped_with_guidance() {
        let out = friendly("database error: disk image is malformed");
        assert!(out.contains("データベース"), "{out}");
        assert!(
            out.contains("disk image is malformed"),
            "detail kept: {out}"
        );
    }

    #[test]
    fn empty_becomes_generic() {
        assert!(friendly("  ").contains("不明な問題"));
    }
}
