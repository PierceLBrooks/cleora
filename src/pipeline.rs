use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::configuration::{Column, Configuration, FileType, OutputFormat, JsonPair};
use crate::embedding::{calculate_embeddings, calculate_embeddings_mmap};
use crate::entity::{EntityProcessor, SMALL_VECTOR_SIZE};
use crate::persistence::embedding::{EmbeddingPersistor, NpyPersistor, TextFileVectorPersistor};
use crate::persistence::entity::InMemoryEntityMappingPersistor;
use crate::sparse_matrix::{create_sparse_matrices, SparseMatrix};
use bus::Bus;
use log::{error, info, warn};
use smallvec::SmallVec;
use std::sync::Arc;
use std::thread;
use rustc_hash::FxHashMap;
use std::sync::mpsc::TryRecvError;

/// Create SparseMatrix'es based on columns config. Every SparseMatrix operates in separate
/// thread. EntityProcessor reads data in main thread and broadcast cartesian products
/// to SparseMatrix'es.
pub fn build_graphs(
    config: &Configuration,
    in_memory_entity_mapping_persistor: Arc<InMemoryEntityMappingPersistor>,
) -> Vec<SparseMatrix> {
    let sparse_matrices = create_sparse_matrices(&config.columns);
    dbg!(&sparse_matrices);

    let mut bus: Bus<SmallVec<[u64; SMALL_VECTOR_SIZE]>> = Bus::new(128);
    let mut bus_weights: Bus<SmallVec<[u64; SMALL_VECTOR_SIZE]>> = Bus::new(192);
    let mut sparse_matrix_threads = Vec::new();
    for mut sparse_matrix in sparse_matrices {
        let mut rx = bus.add_rx();
        let mut rx_weights = bus_weights.add_rx();
        let handle = thread::spawn(move || {
            let mut has_rx = true;
            let mut has_rx_weights = true;
            while has_rx || has_rx_weights {
                if has_rx {
                    match rx.try_recv() {
                        Ok(received) => {
                            sparse_matrix.handle_pair(&received);
                        }
                        Err(TryRecvError::Disconnected) => {
                            has_rx = false;
                        }
                        _ => {}
                    }
                }
                if has_rx_weights {
                    match rx_weights.try_recv() {
                        Ok(received_weights) => {
                            sparse_matrix.handle_weight(&received_weights);
                        }
                        Err(TryRecvError::Disconnected) => {
                            has_rx_weights = false;
                        }
                        _ => {}
                    }
                }
            }
            sparse_matrix.finish();
            sparse_matrix
        });
        sparse_matrix_threads.push(handle);
    }

    for input in config.input.iter() {
        let mut entity_processor = EntityProcessor::new(
            config,
            in_memory_entity_mapping_persistor.clone(),
            |hashes| {
                bus.broadcast(hashes);
            },
            |left, right, weight| {
                let mut arr: SmallVec<[u64; SMALL_VECTOR_SIZE]> = SmallVec::with_capacity(3);
                arr.push(left);
                arr.push(right);
                arr.push(weight);
                bus_weights.broadcast(arr);
            },
            config.log_every_n as u64,
        );

        match &config.file_type {
            FileType::Json => {
                let config_col_num = config.columns.len();
                read_file(input, config.log_every_n as u64, move |line| {
                    let row = parse_json_line(line, &config.columns);
                    let mapping: Vec<SmallVec<[String; SMALL_VECTOR_SIZE]>> = row.iter().map(|v| {
                        v.iter().map(|p| {
                            for outer in p.weights.keys() {
                                for inner in p.weights[outer].keys() {
                                    entity_processor.set_weight(outer, inner, p.weights[outer][inner])
                                }
                            }
                            p.key.clone()
                        }).collect()
                    }).collect();
                    let line_col_num = mapping.len();
                    if line_col_num == config_col_num {
                        entity_processor.process_row(&mapping);
                    } else {
                        warn!("Wrong number of columns (expected: {}, provided: {}). The line [{}] is skipped.", config_col_num, line_col_num, line);
                    }
                });
            }
            FileType::Tsv => {
                let config_col_num = config.columns.len();
                read_file(input, config.log_every_n as u64, move |line| {
                    let row = parse_tsv_line(line);
                    let line_col_num = row.len();
                    if line_col_num == config_col_num {
                        entity_processor.process_row(&row);
                    } else {
                        warn!("Wrong number of columns (expected: {}, provided: {}). The line [{}] is skipped.", config_col_num, line_col_num, line);
                    }
                });
            }
        }
    }

    drop(bus);
    drop(bus_weights);

    let mut sparse_matrices = vec![];
    for join_handle in sparse_matrix_threads {
        let sparse_matrix = join_handle
            .join()
            .expect("Couldn't join on the associated thread");
        sparse_matrices.push(sparse_matrix);
    }

    sparse_matrices
}

/// Read file line by line. Pass every valid line to handler for parsing.
fn read_file<F>(filepath: &str, log_every: u64, mut line_handler: F)
where
    F: FnMut(&str),
{
    let input_file = File::open(filepath).expect("Can't open file");
    let mut buffered = BufReader::new(input_file);

    let mut line_number = 1u64;
    let mut line = String::new();
    loop {
        match buffered.read_line(&mut line) {
            Ok(bytes_read) => {
                // EOF
                if bytes_read == 0 {
                    break;
                }

                line_handler(&line);
            }
            Err(err) => {
                error!("Can't read line number: {}. Error: {}.", line_number, err);
            }
        };

        if line_number % log_every == 0 {
            info!("Number of lines processed: {}", line_number);
        }
        
        // clear to reuse the buffer
        line.clear();

        line_number += 1;
    }
}

/// Parse a line of JSON and read its columns into a vector for processing.
fn parse_json_line(
    line: &str,
    columns: &[Column],
) -> Vec<Vec<JsonPair>> {
    let parsed = json::parse(line).unwrap();
    columns
        .iter()
        .map(|c| {
            let column = &c.name;
            if !c.complex {
                let elem = &parsed[column];
                let value = match *elem {
                    json::JsonValue::String(ref _string) => JsonPair {
                        key: String::from(elem.as_str().unwrap()),
                        weights: FxHashMap::default()
                    },
                    json::JsonValue::Object(ref _object) => if !c.weight { JsonPair {
                        key: String::from(""),
                        weights: FxHashMap::default()
                    } } else { JsonPair {
                        key: String::from(""),
                        weights: FxHashMap::default()
                    } },
                    _ => JsonPair {
                        key: String::from(""),
                        weights: FxHashMap::default()
                    },
                };
                let mut result = Vec::new();
                result.push(value);
                result
            } else {
                parsed
                    [column]
                    .members()
                    .map(|v| match *v {
                        json::JsonValue::String(ref _string) => JsonPair {
                            key: String::from(v.as_str().unwrap()),
                            weights: FxHashMap::default()
                        },
                        json::JsonValue::Object(ref _temp) => if !c.weight { JsonPair {
                            key: String::from(""),
                            weights: FxHashMap::default()
                        } } else {
                            let mut name: String = String::from("");
                            let mut weights: FxHashMap<String, FxHashMap<String, u64>> = FxHashMap::default();
                            for (left, right) in v.entries() {
                                match *right {
                                    json::JsonValue::Object(ref _object) => {
                                        name.push_str(&left);
                                        weights.insert(left.to_string(), FxHashMap::default());
                                        for (outer, inner) in right.entries() {
                                            match *inner {
                                                json::JsonValue::Number(ref _number) => {
                                                    weights.get_mut(&name).unwrap().insert(outer.to_string(), inner.as_u64().unwrap());
                                                },
                                                _ => {}
                                            }
                                        }
                                    },
                                    _ => {}
                                }
                                if name.len() > 0 {
                                    break;
                                }
                            }
                            if name.len() > 0 {
                                JsonPair {
                                    key: name,
                                    weights: weights
                                }
                            } else {
                                JsonPair {
                                    key: String::from(""),
                                    weights: FxHashMap::default()
                                }
                            }
                        },
                        _ => JsonPair {
                            key: String::from(""),
                            weights: FxHashMap::default()
                        },
                    })
                    .collect()
            }
        })
        .collect()
}

/// Parse a line of TSV and read its columns into a vector for processing.
fn parse_tsv_line(line: &str) -> Vec<SmallVec<[&str; SMALL_VECTOR_SIZE]>> {
    let values = line.trim().split('\t');
    values.map(|c| c.split(' ').collect()).collect()
}

/// Train SparseMatrix'es (graphs) in separated threads.
pub fn train(
    config: Configuration,
    in_memory_entity_mapping_persistor: Arc<InMemoryEntityMappingPersistor>,
    sparse_matrices: Vec<SparseMatrix>,
) {
    let config = Arc::new(config);
    let mut embedding_threads = Vec::new();
    for sparse_matrix in sparse_matrices {
        let sparse_matrix = Arc::new(sparse_matrix);
        let config = config.clone();
        let in_memory_entity_mapping_persistor = in_memory_entity_mapping_persistor.clone();
        let handle = thread::spawn(move || {
            let directory = match config.output_dir.as_ref() {
                Some(out) => format!("{}/", out.clone()),
                None => String::from(""),
            };
            let ofp = format!(
                "{}{}__{}__{}.out",
                directory,
                config.relation_name,
                sparse_matrix.col_a_name.as_str(),
                sparse_matrix.col_b_name.as_str()
            );

            let mut persistor: Box<dyn EmbeddingPersistor> = match &config.output_format {
                OutputFormat::TextFile => Box::new(TextFileVectorPersistor::new(
                    ofp,
                    config.produce_entity_occurrence_count,
                )),
                OutputFormat::Numpy => Box::new(NpyPersistor::new(
                    ofp,
                    config.produce_entity_occurrence_count,
                )),
            };
            if config.in_memory_embedding_calculation {
                calculate_embeddings(
                    config.clone(),
                    sparse_matrix.clone(),
                    in_memory_entity_mapping_persistor,
                    persistor.as_mut(),
                );
            } else {
                calculate_embeddings_mmap(
                    config.clone(),
                    sparse_matrix.clone(),
                    in_memory_entity_mapping_persistor,
                    persistor.as_mut(),
                );
            }
        });
        embedding_threads.push(handle);
    }

    for join_handle in embedding_threads {
        join_handle
            .join()
            .expect("Couldn't join on the associated thread");
    }
}
