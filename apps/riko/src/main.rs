use anyhow::{Context, Result};
use clap::Parser;
use riko::{app::App, cli::Cli, run};
use riko_engine::Engine;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let engine = build_engine(&cli).context("building the riko engine")?;
    run::run(App::new(engine)).await
}

fn build_engine(cli: &Cli) -> Result<Engine> {
    let mut builder = Engine::builder();
    if let Some(model) = &cli.model {
        builder = builder.model(model.clone());
    }
    if let Some(workspace) = &cli.workspace {
        builder = builder.workspace_root(workspace.clone());
    }
    if let Some(session) = &cli.session {
        builder = builder.session_path(session.clone());
    }
    Ok(builder.build()?)
}
