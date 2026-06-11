use smol_str::SmolStr;
use uuid::Uuid;

/// Fresh UUID v4 rendered as a hyphenated `SmolStr` (36 chars).
pub fn uuid_v4() -> SmolStr {
    SmolStr::from(Uuid::new_v4().to_string())
}

/// Fresh 12-hex-char id sliced from a UUID v4 — enough entropy for in-session uniqueness.
pub fn short_id() -> SmolStr {
    let hex = Uuid::new_v4().simple().to_string();
    SmolStr::from(&hex[..12])
}

/// Define an opaque, prefixed, `SmolStr`-backed identifier.
///
/// The generated type is serde-transparent and `Clone + Eq + Hash + Display`. `fresh`
/// mints a random id carrying the given prefix; `new`/`From<&str>` wrap an existing value.
#[macro_export]
macro_rules! id_type {
    ($(#[$meta:meta])* $Name:ident, $prefix:literal) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, PartialEq, Eq, Hash, ::serde::Serialize, ::serde::Deserialize,
        )]
        #[serde(transparent)]
        pub struct $Name(::smol_str::SmolStr);

        impl $Name {
            /// Wrap an existing string value without validation.
            pub fn new(value: impl ::core::convert::Into<::smol_str::SmolStr>) -> Self {
                Self(value.into())
            }

            /// Mint a fresh id carrying this type's prefix.
            pub fn fresh() -> Self {
                Self(::smol_str::SmolStr::from(::std::format!(
                    "{}_{}",
                    $prefix,
                    $crate::short_id()
                )))
            }

            /// Borrow the underlying string.
            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }

            /// Consume the id and return the inner string.
            pub fn into_inner(self) -> ::smol_str::SmolStr {
                self.0
            }
        }

        impl ::core::fmt::Display for $Name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.write_str(self.0.as_str())
            }
        }

        impl ::core::convert::From<&str> for $Name {
            fn from(value: &str) -> Self {
                Self(::smol_str::SmolStr::from(value))
            }
        }
    };
}
