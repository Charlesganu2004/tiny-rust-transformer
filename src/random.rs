//! Tiny deterministic random-number generator, so we need no external crates.
//! Good for learning, sampling, and weight initialization; NOT cryptography.

pub struct Random {
    state: u64,
}

impl Random {
    pub fn new(seed: u64) -> Self {
        // Xorshift gets stuck at 0, so replace zero seeds.
        Self { state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed } }
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        // Value in [0,1) with 24 random bits of precision.
        (self.next_u32() >> 8) as f32 / (1_u32 << 24) as f32
    }
}
