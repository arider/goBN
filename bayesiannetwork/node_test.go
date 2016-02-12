package bayesiannetwork

import (
	"math/rand"
	"testing"
)

func TestDensitySample(t *testing.T) {
	source := rand.NewSource(1)
	r := rand.New(source)
	network := initStudentNetwork()
	for i := 0; i < 10; i++ {
		// check against the Node with 3 States
		index := network.Nodes[4].cpd[0].sample(r)
		if index >= len(network.Nodes[4].cpd[1].index) {
			t.Fail()
		}
	}
}

func TestGetCPDIndex(t *testing.T) {

	network := initStudentNetwork()
	network.topologicalSort()
	D := network.Nodes[0]
	I := network.Nodes[1]
	G := network.Nodes[3]

	// I is index 3, D is index 2; States 0, 1 should correspond to element 2
	// the parent indices in rows:
	// 0 0
	// 1 0
	// 0 1 *
	// 1 1
	nodeStates := make(map[*Node]int)
	nodeStates[I] = 0
	nodeStates[D] = 0
	if G.getCPDIndex(nodeStates) != 0 {
		t.Fail()
	}

	nodeStates[I] = 0
	nodeStates[D] = 1
	if G.getCPDIndex(nodeStates) != 2 {
		t.Fail()
	}

	nodeStates[I] = 1
	nodeStates[D] = 0
	if G.getCPDIndex(nodeStates) != 1 {
		t.Fail()
	}

	nodeStates[I] = 1
	nodeStates[D] = 1
	if G.getCPDIndex(nodeStates) != 3 {
		t.Fail()
	}
}

func TestGetAllParentStates(t *testing.T) {
	network := initStudentNetwork()
	network.topologicalSort()

	// Node G in the network -- 2 parents, 6 States
	n := network.Nodes[3]

	nStates := 1
	parentStateNums := make([]int, 0)
	for _, p := range n.Parents {
		nStates *= p.States
		parentStateNums = append(parentStateNums, p.States)
	}
	current := make([]int, 0)
	States := make([][]int, 0)
	n.getAllParentStates(&States, current, parentStateNums...)

	if len(States) != 4 {
		t.Fail()
	}
	if States[1][0] != 1 && States[1][1] != 0 {
		t.Fail()
	}
}
