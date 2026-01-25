use std::io::{self, BufRead};
use std::fs::File;
use std::path::PathBuf;
use rust_engine::eval;
use chess::Board;
use std::str::FromStr;

fn main() -> io::Result<()> {
    // Basic Texel Tuner skeleton
    // 1. Load EPD positions
    // 2. Calculate error (Sigmoid(eval) - result)^2
    // 3. This script just calculates total error for now

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(manifest_dir).join("tuning_data.epd");

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(_) => {
            println!("Could not open {}. Run generate_tuning.py first.", path.display());
            return Ok(());
        }
    };

    let reader = io::BufReader::new(file);
    let mut total_error = 0.0;
    let mut count = 0;

    // K factor for scaling eval to probability
    let k = 1.13; // Usually determined empirically

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }

        // Simple EPD parse: "FEN c9 "1.0";"
        let parts: Vec<&str> = line.split(" c9 ").collect();
        if parts.len() < 2 { continue; }

        let fen = parts[0];
        let result_str = parts[1].replace("\"", "").replace(";", "").trim().to_string();
        let result: f64 = result_str.parse().unwrap_or(0.5);

        if let Ok(board) = Board::from_str(fen) {
            let eval_score = eval::evaluate(&board) as f64;

            // Sigmoid: 1 / (1 + 10^(-K * eval / 400))
            let sigmoid = 1.0 / (1.0 + 10.0_f64.powf(-k * eval_score / 400.0));

            let error = (result - sigmoid).powi(2);
            total_error += error;
            count += 1;
        }
    }

    if count > 0 {
        println!("Total Error: {:.6}", total_error);
        println!("Average Error: {:.6}", total_error / count as f64);
    } else {
        println!("No valid positions found.");
    }

    Ok(())
}
