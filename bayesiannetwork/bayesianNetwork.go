package bayesiannetwork

import (
	"geneticalgorithm"
	"math"
	"math/rand"
	"sync"
)

// The net is a DAG of Node where the CPD of a Node is ordered by parent
// order
type BayesianNetwork struct {
	Nodes []*Node
}

// initialize a bayesian net
func NewBayesianNetwork() *BayesianNetwork {
	return &BayesianNetwork{Nodes: make([]*Node, 0)}
}

// add a directed edge between Nodes p and c
func (net BayesianNetwork) AddEdge(p, c *Node) {
	c.Parents = append(c.Parents, p)
	p.Children = append(p.Children, c)
}

// Sample from the bayesian net returns a []int of the
// original index sampled from all Nodes in the net in order
// that the Nodes appear in the net
func (net BayesianNetwork) Sample(r *rand.Rand) (sample map[*Node]int) {
	sample = make(map[*Node]int)
	for _, n := range net.Nodes {
		parent_sample := make(map[*Node]int)
		if len(n.Parents) == 0 {
			sample[n] = n.cpd[0].sample(r)
		}
		for _, parent := range n.Parents {
			parent_sample[parent] = sample[parent]
		}
		cpdIndex := n.getCPDIndex(parent_sample)

		sample[n] = n.cpd[cpdIndex].sample(r)
	}

	return sample
}

// Kahn algorithm for topological sort -- returns whether the net has
// cycles.  Updates the Node order in the net regardless of presence of
// cycles.
func (net *BayesianNetwork) topologicalSort() (hasCycles bool) {
	var sorted, rootNodes []*Node

	// find the Nodes with no Parents and count the number of edges to each
	// Node
	remaining := make(map[*Node]int)
	for _, n := range net.Nodes {
		if len(n.Parents) == 0 {
			rootNodes = append(rootNodes, n)
		} else {
			remaining[n] = len(n.Parents)
		}
	}

	for len(rootNodes) > 0 {
		// pop a Node from rootNodes
		last := len(rootNodes) - 1
		n := rootNodes[last]
		rootNodes = rootNodes[:last]

		// push n to sorted
		sorted = append(sorted, n)
		for _, m := range n.Children {
			if remaining[m] > 0 {
				remaining[m]--
				if remaining[m] == 0 {
					rootNodes = append(rootNodes, m)
				}
			}
		}
	}
	// check for cycles
	for n, in := range remaining {
		if in > 0 {
			for _, child := range n.Children {
				if remaining[child] > 0 {
					hasCycles = true
					break
				}
			}
		}
	}
	// reorder the net regardless of cycles
	net.Nodes = sorted
	return hasCycles
}

// Tarjan's algorithm for topological sort
func (net *BayesianNetwork) topologicalSortTarjan() {
	if HasCycles(net) {
		panic("Network has cycles, can't topological sort.")
	}

	//	fmt.Println("TESTING TOP")
	sorted := make([]*Node, 0, len(net.Nodes))
	visited := make(map[*Node]bool)

	for _, n := range net.Nodes {
		if _, exists := visited[n]; !exists {
			visit(n, visited, &sorted)
		}
	}

	// reverse the order so that it's Parents first
	reverse := make([]*Node, len(sorted))
	for i, j := len(sorted)-1, 0; i >= 0; i, j = i-1, j+1 {
		reverse[j] = sorted[i]
	}
	net.Nodes = reverse
}

// The DFS part of Tarjan's algorithm for topological sort
func visit(n *Node, visited map[*Node]bool, sorted *[]*Node) {
	if _, exists := visited[n]; !exists {
		for _, child := range n.Children {
			visit(child, visited, sorted)
		}
		visited[n] = true
		*sorted = append(*sorted, n)
	}
}

// Sample the net randomly to build up a set of evidence to query; assumes
// the net is in topological order
func (net BayesianNetwork) logicSampling(n_samples int, r *rand.Rand) []map[*Node]int {
	var wg sync.WaitGroup

	samples := make([]map[*Node]int, n_samples)
	for s := 0; s < n_samples; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			samples[s] = net.Sample(r)
		}()
		wg.Wait()
	}

	return samples
}

// calculate the probabiliy of a slice of Node States given a
// map of evidence States
func (net BayesianNetwork) PosteriorDistribution(
	r *rand.Rand,
	dependent []*Node,
	evidence map[*Node]int) map[*Node]Density {

	// TODO: this is a bad heuristic
	samples := net.logicSampling(len(net.Nodes)*100, r)

	// collect the samples that meet the evidence
	keep := make([]int, 0)
	for i, sample := range samples {
		pass := true
		for node, state := range evidence {
			if sample[node] != state {
				pass = false
			}
		}
		if pass {
			keep = append(keep, i)
		}
	}

	// load remaining with the samples that pass all filters
	remaining := make([]map[*Node]int, 0)
	for _, sample_index := range keep {
		remaining = append(remaining, samples[sample_index])
	}

	// determine the distribution of all dependent Nodes based on the remaining
	// samples
	densities := make(map[*Node]Density, 0)
	for _, Node := range dependent {
		observed := make([]int, Node.States)
		count := 0
		for _, sample := range remaining {
			observed[sample[Node]] += 1
			count += 1
		}
		probs := make([]float64, Node.States)
		for i := 0; i < len(observed); i++ {
			probs[i] = float64(observed[i]) / float64(count)
		}
		densities[Node] = NewDensity(probs...)
	}
	return densities
}

// Calculate the model Likelihood for each of a set of
// observations
func (net BayesianNetwork) Likelihood(observations []map[*Node]int) []float64 {
	likelihoods := make([]float64, len(observations))

	var wg sync.WaitGroup
	for index, instance := range observations {
		likelihoods[index] = 1.0

		wg.Add(1)
		go func() {
			defer wg.Done()

			for _, n := range net.Nodes {
				State := instance[n]
				parentStates := make(map[*Node]int)
				for _, parent := range n.Parents {
					parentStates[parent] = instance[parent]
				}
				cpd_index := n.getCPDIndex(parentStates)
				likelihoods[index] *= n.cpd[cpd_index].StateMap[State]
			}
		}()
		wg.Wait()
	}

	return likelihoods
}

func (net BayesianNetwork) ModelLikelihood(observations []map[*Node]int) float64 {
	Likelihoods := net.Likelihood(observations)
	ml := 0.0
	for _, i := range Likelihoods {
		ml += math.Log(i)
	}
	return ml
}

// take a slice of observed Node States and update the CPDs of all Nodes
func (net *BayesianNetwork) updateWeights(observations []map[*Node]int) {
	// assuming that the Nodes are in topological order
	net.topologicalSort()

	for _, n := range net.Nodes {
		// get all unique parent States from the observations
		parentStateNums := make([]int, 0)
		for _, p := range n.Parents {
			parentStateNums = append(parentStateNums, p.States)
		}

		current := make([]int, 0)
		States := make([][]int, 0)
		n.getAllParentStates(&States, current, parentStateNums...)
		n.cpd = make([]Density, len(States))

		// for each unique parent State, filter out the observations that
		// match and update the CPD
		for _, State := range States {
			// get the cpd index of this State
			nodeStates := make(map[*Node]int)
			for i, p := range n.Parents {
				nodeStates[p] = State[i]
			}
			cpdIndex := n.getCPDIndex(nodeStates)

			// filter out the observations that don't match this State
			total := 0.0
			sums := make([]float64, n.States)
			for _, observation := range observations {
				pass := true
				for n, s := range nodeStates {
					if observation[n] != s {
						pass = false
					}
				}
				if pass {
					sums[observation[n]] += 1.0
					total += 1.0
				}
			}
			// calculate the probabilities
			for i := 0; i < len(sums); i++ {
				sums[i] = sums[i] / total
			}
			n.cpd[cpdIndex] = NewDensity(sums...)
		}
	}
}

// given observations, update the States and CPDs of the Nodes
func (net *BayesianNetwork) inferNodeStates(data []map[*Node]int) {
	nodes := make([]*Node, 0)
	nodeStates := make(map[*Node]int)

	max := func(a int, b int) int {
		if b > a {
			return b
		}
		return a
	}
	// get the number of States in the data set for each Node
	for _, instance := range data {
		for n, v := range instance {
			nodeStates[n] = max(nodeStates[n], v)
		}
	}
	// 0 is a State, length needs to be + 1
	for n, _ := range nodeStates {
		nodeStates[n] += 1
	}

	// update the Nodes
	for n, v := range nodeStates {
		nodes = append(nodes, n)
		n.States = v
	}

	net.Nodes = nodes
}

// infer a bayesian net from a slice of map[*Node]int Node States
func InferBayesianNetwork(data []map[*Node]int, iterations int) *BayesianNetwork {

	// set the order that the Nodes appear in the binary representation
	// just assume that the first instance has all Nodes
	nodeOrder := make([]*Node, 0)
	for n, _ := range data[0] {
		nodeOrder = append(nodeOrder, n)
	}

	net := NewBayesianNetwork()
	net.inferNodeStates(data)

	mutateCheckFunction := func(edges []int) (ok bool) {
		// make a net with the given topology
		net.binaryToTopology(nodeOrder, edges)

		// check for cycles
		isDAG := !HasCycles(net)
		return isDAG
	}

	// TODO: overfitting; use random subset for evaluation
	scoreFunction := func(bits []int) float64 {
		net.binaryToTopology(nodeOrder, bits)
		net.inferNodeStates(data)
		net.updateWeights(data)
		return net.ModelLikelihood(data)
	}

	best := geneticalgorithm.Evolve(
		scoreFunction,
		mutateCheckFunction,
		len(data[0])*len(data[0]),
		iterations,
		false)

	// the net is updated on every scoreFunction call, so we
	// have to update to use the best topology here
	net.binaryToTopology(nodeOrder, best)
	net.inferNodeStates(data)
	net.updateWeights(data)

	return net
}

// TODO: Likelihood weighting
// TODO: AIS-BN algorithm, see Cheng and Druzdzel, AAAI, 2000
