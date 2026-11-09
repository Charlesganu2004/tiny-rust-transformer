//! Tiny reverse-mode automatic differentiation (autograd).
//! Every math operation records its value and how to pass gradients backward.

// A Var is an index into Tape.nodes, not the number itself.
#[derive(Clone, Copy, Debug)]
pub struct Var(pub usize);

// An Edge means: parent_variable receives gradient * local_derivative.
// Two slots are enough because every operation below has at most two inputs.
#[derive(Clone, Copy)]
struct Node {
    value: f32,                 // The number computed in the forward pass.
    grad: f32,                  // d(loss)/d(this node), filled in backward().
    edges: [(usize, f32); 2],   // (parent_index, local_derivative).
    edge_count: usize,          // 0, 1, or 2 active edges.
}

pub struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    // Store a constant or trainable weight in the computation graph.
    pub fn leaf(&mut self, value: f32) -> Var {
        self.push(value, &[])
    }

    // All operations call this to append a new node to the graph.
    fn push(&mut self, value: f32, edges: &[(Var, f32)]) -> Var {
        let mut node = Node {
            value,
            grad: 0.0,
            edges: [(0, 0.0); 2],
            edge_count: edges.len(),
        };
        for (i, (parent, derivative)) in edges.iter().enumerate() {
            node.edges[i] = (parent.0, *derivative);
        }
        let index = self.nodes.len();
        self.nodes.push(node);
        Var(index)
    }

    pub fn value(&self, x: Var) -> f32 {
        self.nodes[x.0].value
    }

    pub fn grad(&self, x: Var) -> f32 {
        self.nodes[x.0].grad
    }

    // If z = a + b, then dz/da = 1 and dz/db = 1.
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.push(self.value(a) + self.value(b), &[(a, 1.0), (b, 1.0)])
    }

    // If z = a - b, then dz/da = 1 and dz/db = -1.
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        self.push(self.value(a) - self.value(b), &[(a, 1.0), (b, -1.0)])
    }

    // Product rule: d(a*b)/da = b, and d(a*b)/db = a.
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        self.push(self.value(a) * self.value(b), &[(a, self.value(b)), (b, self.value(a))])
    }

    // Multiply by an ordinary floating point number (not a trainable weight).
    pub fn scale(&mut self, x: Var, factor: f32) -> Var {
        self.push(self.value(x) * factor, &[(x, factor)])
    }

    // Quotient rule. The denominator is expected to be positive for our use.
    pub fn div(&mut self, a: Var, b: Var) -> Var {
        let numerator = self.value(a);
        let denominator = self.value(b);
        self.push(
            numerator / denominator,
            &[(a, 1.0 / denominator), (b, -numerator / (denominator * denominator))],
        )
    }

    // The derivative of exp(x) is exp(x).
    pub fn exp(&mut self, x: Var) -> Var {
        let result = self.value(x).exp();
        self.push(result, &[(x, result)])
    }

    // The derivative of ln(x) is 1/x.
    pub fn ln(&mut self, x: Var) -> Var {
        self.push(self.value(x).ln(), &[(x, 1.0 / self.value(x))])
    }

    // ReLU(x) = max(0, x); gradient is 1 when positive, otherwise 0.
    pub fn relu(&mut self, x: Var) -> Var {
        let v = self.value(x);
        self.push(v.max(0.0), &[(x, if v > 0.0 { 1.0 } else { 0.0 })])
    }

    // Reverse traversal is the chain rule: gradients flow from loss to weights.
    pub fn backward(&mut self, loss: Var) {
        self.nodes[loss.0].grad = 1.0; // d(loss)/d(loss) = 1.
        for i in (0..self.nodes.len()).rev() {
            let gradient_here = self.nodes[i].grad;
            let edge_count = self.nodes[i].edge_count;
            for edge in 0..edge_count {
                let (parent, local_derivative) = self.nodes[i].edges[edge];
                self.nodes[parent].grad += gradient_here * local_derivative;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_rule_works() {
        let mut tape = Tape::new();
        let x = tape.leaf(3.0);
        let y = tape.leaf(4.0);
        let product = tape.mul(x, y);
        tape.backward(product);
        assert_eq!(tape.value(product), 12.0);
        assert_eq!(tape.grad(x), 4.0);
        assert_eq!(tape.grad(y), 3.0);
    }

    #[test]
    fn shared_node_accumulates_gradients() {
        let mut tape = Tape::new();
        let x = tape.leaf(3.0);
        let square = tape.mul(x, x);
        tape.backward(square);
        assert_eq!(tape.grad(x), 6.0);
    }
}