package bayesiannetwork

import (
	"math"
	"sort"
	"strconv"
	"testing"
)

func TestSwap(t *testing.T) {
	ar := NewIntIndexSlice(5, 4, 3, 2, 1)
	sort.Sort(ar)

	sorted, index := ar.Interface.(sort.IntSlice), ar.index

	solution := [5]int{4, 3, 2, 1, 0}
	worked := true
	for i := 0; i < len(ar.index)-1; i++ {
		if sorted[i] >= sorted[i+1] || index[i] != solution[i] {
			worked = false
		}
	}
	if !worked {
		t.Fail()
	}
}

func TestHasCycles(t *testing.T) {
	network := NewBayesianNetwork()
	A := Node{Name: "A"}
	B := Node{Name: "B"}
	C := Node{Name: "C"}
	network.Nodes = []*Node{&A, &B, &C}
	network.AddEdge(&A, &B)
	network.AddEdge(&B, &C)

	// test with no cycles
	results := HasCycles(network)
	if results {
		t.Fail()
	}

	// test with cycles
	network.AddEdge(&C, &A)
	results = HasCycles(network)
	if !results {
		t.Fail()
	}

	// test self edge
	network = NewBayesianNetwork()
	network.Nodes = []*Node{&A}
	A = Node{Name: "A"}
	A.Children = make([]*Node, 0)
	A.Parents = make([]*Node, 0)
	network.AddEdge(&A, &A)

	results = HasCycles(network)
	if !results {
		t.Fail()
	}

}

func makeCyclicalNetwork() BayesianNetwork {

	network := NewBayesianNetwork()
	A := Node{Name: "A"}
	B := Node{Name: "B"}
	C := Node{Name: "C"}
	D := Node{Name: "D"}
	E := Node{Name: "E"}
	network.Nodes = []*Node{&A, &B, &C, &D, &E}
	network.AddEdge(&A, &B)
	network.AddEdge(&B, &C)
	network.AddEdge(&C, &A)
	network.AddEdge(&D, &A)
	network.AddEdge(&D, &E)
	network.AddEdge(&E, &A)

	return *network
}

func TestNetworkToBinary(t *testing.T) {
	network := makeCyclicalNetwork()
	solution := []int{
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		1, 0, 0, 0, 0,
		1, 0, 0, 0, 1,
		1, 0, 0, 0, 0}

	result := networkToBinary(network.Nodes)
	for i := 0; i < len(solution); i++ {
		if solution[i] != result[i] {
			t.Fail()
		}
	}
}

func TestBinaryToTopology(t *testing.T) {
	solution := []int{
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		1, 0, 0, 0, 0,
		1, 0, 0, 0, 1,
		1, 0, 0, 0, 0}

	Nodes := make([]*Node, 5)
	// create the Nodes
	nNodes := int(math.Sqrt(float64(len(solution))))

	network := BayesianNetwork{}
	for i := 0; i < len(solution); i++ {
		if i%nNodes == 0 {
			Nodes[i/nNodes] = &Node{Name: strconv.Itoa(i / nNodes)}
		}
	}
	network.Nodes = Nodes
	network.binaryToTopology(network.Nodes, solution)

	result := networkToBinary(network.Nodes)
	for i := 0; i < len(solution); i++ {
		//		t.Log(i, solution[i], result[i])
		if solution[i] != result[i] {
			t.Fail()
		}
	}
}

func TestConvertDataset(t *testing.T) {

	data := make([][]int, 0)
	data = append(data, []int{0, 0, 0})
	data = append(data, []int{1, 0, 0})
	data = append(data, []int{0, 1, 0})
	data = append(data, []int{0, 0, 1})
	data = append(data, []int{1, 1, 1})

	featureNames := []string{"A", "B", "C"}

	converted := ConvertDataset(data, featureNames)
	nodes := make([]*Node, 0)
	for Node, _ := range converted[0] {
		nodes = append(nodes, Node)
	}

	sum := func(array []int) int {
		out := 0
		for _, e := range array {
			out += e
		}
		return out
	}
	sumHash := func(hash map[*Node]int) int {
		out := 0
		for _, e := range hash {
			out += e
		}
		return out
	}
	if sum(data[0]) != sumHash(converted[0]) {
		t.Fail()
	}
	if sum(data[1]) != sumHash(converted[1]) {
		t.Fail()
	}
	if sum(data[2]) != sumHash(converted[2]) {
		t.Fail()
	}
	if sum(data[3]) != sumHash(converted[3]) {
		t.Fail()
	}
	if sum(data[4]) != sumHash(converted[4]) {
		t.Fail()
	}
}
