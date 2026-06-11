riko_utils::id_type!(ToolCallId, "tc");

riko_utils::id_type!(ModelRef, "mdl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_match_type() {
        assert!(ToolCallId::fresh().as_str().starts_with("tc_"));
        assert!(ModelRef::fresh().as_str().starts_with("mdl_"));
    }
}
