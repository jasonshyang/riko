use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use riko_core::{Result, RikoError};
use serde::{Deserialize, Serialize};

use crate::{BranchId, Operation};

const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Header {
    version: u32,
    root: BranchId,
}

/// Append-only writer for a workspace's operation log.
pub(crate) struct OperationSink {
    writer: BufWriter<File>,
}

impl OperationSink {
    /// Create a fresh log file and write its header. Fails if the file already exists.
    pub(crate) fn create(path: &Path, root: BranchId) -> Result<Self> {
        let file = OpenOptions::new().create_new(true).write(true).open(path)?;
        let mut sink = Self { writer: BufWriter::new(file) };
        sink.write_line(&Header { version: FORMAT_VERSION, root })?;
        Ok(sink)
    }

    /// Open an existing log file for appending.
    pub(crate) fn open_existing(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self { writer: BufWriter::new(file) })
    }

    pub(crate) fn append(&mut self, op: &Operation) -> Result<()> {
        self.write_line(op)
    }

    fn write_line<T: Serialize>(&mut self, value: &T) -> Result<()> {
        serde_json::to_writer(&mut self.writer, value)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
}

/// Read a workspace log: the root branch id from the header, then every logged operation in
/// order. The caller replays the operations to rebuild state.
pub(crate) fn read_log(path: &Path) -> Result<(BranchId, Vec<Operation>)> {
    let mut lines = BufReader::new(File::open(path)?).lines();
    let header_line = match lines.next() {
        Some(line) => line?,
        None => return Err(RikoError::Config("session file is empty".into())),
    };
    let header: Header = serde_json::from_str(&header_line)?;
    if header.version != FORMAT_VERSION {
        return Err(RikoError::Config(format!(
            "unsupported session format version {}",
            header.version
        )));
    }
    let mut ops = Vec::new();
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        ops.push(serde_json::from_str(&line)?);
    }
    Ok((header.root, ops))
}
