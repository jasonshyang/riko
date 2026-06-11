pub mod bash;
pub mod edit;
pub mod find;
pub mod grep;
pub mod ls;
pub mod read;
pub mod write;

/// Build the wire JSON-Schema for a `#[derive(JsonSchema)]` argument type. Goes through
/// `SchemaGenerator` so it can stay generic over `T` (the `schema_for!` macro needs a concrete
/// type at the call site).
///
/// Infallible: a schemars [`Schema`](schemars::Schema) already wraps a `serde_json::Value`, so
/// extracting it is a move, not a fallible serialization.
fn schema_for<T: schemars::JsonSchema>() -> riko_core::ToolSchema {
    let schema = schemars::SchemaGenerator::default().into_root_schema_for::<T>();
    riko_core::ToolSchema::from_value(schema.to_value())
}
