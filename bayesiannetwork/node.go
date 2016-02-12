package bayesiannetwork

import (
	"math/rand"
	"sort"
)

// Hold the PMF in sorted order by probabilty and the original indices of the
// PMF (in "histogram order," I suppose)
type Density struct {
	sorted []float64
	// index is the original indices of the sorted probabilities -- corresponds
	// to Node State
	index     []int
	prefixSum []float64
	// index in reverse -- maps Node State to probability
	StateMap map[int]float64
}

func NewDensity(probs ...float64) Density {
	slice := NewFloat64IndexSlice(probs...)
	StateMap := make(map[int]float64)
	for k, p := range probs {
		StateMap[k] = p
	}

	sort.Sort(slice)
	sorted, index := slice.Interface.(sort.Float64Slice), slice.index

	prefixSum := make([]float64, len(sorted))
	current := 0.0
	for i, v := range sorted {
		current += v
		prefixSum[i] = current
	}
	d := Density{sorted, index, prefixSum, StateMap}
	return d
}

// Densitys hold the sorted probailities and the original indices
// the sample function will return the original index of the "bucket" sampled.
func (d Density) sample(r *rand.Rand) int {
	index := sort.SearchFloat64s(d.prefixSum, r.Float64())
	//	fmt.Println("CHECK", len(d.sorted), index)
	return d.index[index]
}

// Basic Node in a bayesian network.  Allows only discrete distributions. All
// cpds must have the same size.
type Node struct {
	Name     string
	States   int
	Children []*Node
	Parents  []*Node
	cpd      []Density
}

func (n Node) getCPDIndex(parent_States map[*Node]int) int {

	if len(n.Parents) != len(parent_States) {
		return -1
	}
	parent_cpd_sizes := make([]int, len(n.Parents))

	cpd_length := 1
	for index, parent := range n.Parents {
		cpd_length = cpd_length * parent.States
		parent_cpd_sizes[index] = parent.States
	}

	start := 0
	end := cpd_length
	//	running_length := cpd_length
	for i := len(n.Parents) - 1; i >= 0; i-- {
		repeats := (end - start) / parent_cpd_sizes[i]
		start = start + parent_States[n.Parents[i]]*repeats
		end = start + repeats
	}
	return start

}

// function to enumerate all possible parent States
func (n Node) getAllParentStates(states *[][]int, current []int, parentStateNums ...int) {
	if len(parentStateNums) == 0 {
		*states = append(*states, current)
	} else {
		for s := parentStateNums[0] - 1; s >= 0; s-- {
			next := append(current, s)
			n.getAllParentStates(states, next, parentStateNums[1:]...)
		}
	}
}
