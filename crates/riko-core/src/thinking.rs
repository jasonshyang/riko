/// How much a model should think before answering. Each provider clamps this to the
/// nearest level it actually supports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThinkingLevel {
    /// No extended thinking.
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    /// Maximum thinking budget the provider supports.
    Max,
}

impl ThinkingLevel {
    /// Lowercase identifier matching the serde wire format.
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingLevel::Off => "off",
            ThinkingLevel::Minimal => "minimal",
            ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
            ThinkingLevel::Max => "max",
        }
    }
}

impl std::fmt::Display for ThinkingLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
