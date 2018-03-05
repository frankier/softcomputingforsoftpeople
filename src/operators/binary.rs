use rand::Rng;
//use std::mem::transmute;
use safe_transmute::{PodTransmutable, guarded_transmute_to_mut_bytes_pod};
use bitwise::SetBitsGeq;
// guarded_transmute_to_bytes_pod, 


/*fn gene_bits(gene: Gene) -> u16 {
    unsafe { transmute::<Gene, u16>(gene) }
}

fn gene_bits_mut(gene: &mut Gene) -> &mut u16 {
    unsafe { transmute::<&mut Gene, &mut u16>(gene) }
}

fn and(a: &[u8], b: &[u8]) -> Box<[u8]> {
    assert!(a.len() == b.len());
    let c = Vec::with_capacity(a.len());
    for (&a_i, &b_i) in a.iter().zip(b.iter()) {
        c.push(a_i & b_i);
    }
    c.into_boxed_slice()
}

fn xor_assign(a: &mut [u8], b: &[u8]) {
    assert!(a.len() == b.len());
    for (&mut a_i, &b_i) in a.iter_mut().zip(b.iter()) {
        a_i ^= b_i;
    }
}
*/

/// Performs single-point crossover in place, replacing the parents with the children
pub fn crossover_inplace<G: PodTransmutable>(mummy: &mut G, daddy: &mut G, bit: u8) {
    let mummys_bits = guarded_transmute_to_mut_bytes_pod(mummy);
    let daddys_bits = guarded_transmute_to_mut_bytes_pod(daddy);
    let xover_idx = (bit / 8) as usize;
    let (daddys_bits_initial, daddys_bits_rest) = daddys_bits.split_at_mut(xover_idx);
    let (mummys_bits_initial, mummys_bits_rest) = mummys_bits.split_at_mut(xover_idx);
    daddys_bits_initial.swap_with_slice(mummys_bits_initial);
    let d = &mut daddys_bits_rest[0];
    let m = &mut mummys_bits_rest[0];
    {
        let sub_bit = bit % 8;
        let mask = 0_u8.set_bits_geq(sub_bit);
        *d ^= *m & mask;
        *m ^= *d & mask;
        *d ^= *m & mask;
    }
}

pub fn crossover<G: PodTransmutable>(mut mummy: G, mut daddy: G, bit: u8) -> (G, G) {
    crossover_inplace(&mut mummy, &mut daddy, bit);
    (mummy, daddy)
}

pub fn mutate_inplace<R: Rng, G: PodTransmutable>(rng: &mut R, parent: &mut G) {
    // XXX:
    let bytes = guarded_transmute_to_mut_bytes_pod(parent);
    let flip_prob = 1.0 / (bytes.len() * 8) as f32;
    for p_i in bytes.iter_mut() {
        let mut mask = 1;
        for _ in 0..8 {
            let flip_chance: f32 = rng.gen();
            if flip_chance < flip_prob {
                *p_i ^= mask;
            }
            mask <<= 1;
        }
    }
}
