package bayesiannetwork

import (
	"math"
	"math/rand"
	"testing"
)

func initStudentNetwork() *BayesianNetwork {
	network := NewBayesianNetwork()
	I := Node{Name: "I", cpd: []Density{NewDensity(.7, .3)}, States: 2}
	D := Node{Name: "D", cpd: []Density{NewDensity(.6, .4)}, States: 2}
	// the order of G's densities is changed from the example to reflect how
	// CPD indices are calculated in Node.go
	G := Node{Name: "G",
		cpd: []Density{
			NewDensity(.3, .4, .3),
			NewDensity(.9, .08, .02),
			NewDensity(.05, .25, .7),
			NewDensity(.5, .3, .2)}, States: 3}
	S := Node{Name: "S",
		cpd: []Density{
			NewDensity(.95, .05),
			NewDensity(.2, .8)}, States: 2}
	L := Node{Name: "L",
		cpd: []Density{
			NewDensity(.1, .9),
			NewDensity(.4, .6),
			NewDensity(.99, .01)}, States: 2}

	network.Nodes = []*Node{&I, &S, &L, &D, &G}

	network.AddEdge(&I, &G)
	network.AddEdge(&D, &G)
	network.AddEdge(&I, &S)
	network.AddEdge(&G, &L)

	return network
}

func TestInitStudentNetwork(t *testing.T) {
	//	network := initStudentNetwork()
	//	t.Log("NETWORK: ")
	//	for _, n := range network.Nodes {
	//		t.Log(*n)
	//	}
}

func TestTopologicalSort(t *testing.T) {
	network := initStudentNetwork()
	network.topologicalSort()

	// check that no child is listed before it's parent
	listed := make(map[string]bool)
	for _, n := range network.Nodes {
		listed[n.Name] = true
		for _, child := range n.Children {
			if _, exists := listed[child.Name]; exists {
				t.Fail()
			}
		}
	}

	// test self edge
	network = NewBayesianNetwork()
	A := Node{Name: "A"}
	network.Nodes = []*Node{&A}
	A.Children = make([]*Node, 0)
	A.Parents = make([]*Node, 0)
	network.AddEdge(&A, &A)
	detected := network.topologicalSort()
	if !detected {
		t.Fail()
	}
}

func TestTopologicalSortTarjan(t *testing.T) {
	network := initStudentNetwork()
	network.topologicalSortTarjan()

	// check that no child is listed before it's parent
	listed := make(map[string]bool)
	for _, n := range network.Nodes {
		listed[n.Name] = true
		for _, child := range n.Children {
			if _, exists := listed[child.Name]; exists {
				t.Fail()
			}
		}
	}

	// test self edge
	network = NewBayesianNetwork()
	A := Node{Name: "A"}
	network.Nodes = []*Node{&A}
	A.Children = make([]*Node, 0)
	A.Parents = make([]*Node, 0)
	network.AddEdge(&A, &A)
	defer func() {
		// fail if the next call to topologicalSort doesn't panic
		if r := recover(); r == nil {
			t.Fail()
		}
	}()
	network.topologicalSortTarjan()
}

func TestBayesianNetworkSample(t *testing.T) {
	network := initStudentNetwork()
	network.topologicalSort()
	source := rand.NewSource(1)
	r := rand.New(source)

	pass := false
	previous := network.Sample(r)
	for i := 0; i < 100; i++ {
		sample := network.Sample(r)
		for i, _ := range sample {
			if previous[i] != sample[i] {
				pass = true
				break
			}
		}
	}
	if !pass {
		t.Fail()
	}
}

func TestLogicSampling(t *testing.T) {
	source := rand.NewSource(1)
	r := rand.New(source)

	network := initStudentNetwork()
	network.topologicalSort()

	out := network.logicSampling(10, r)
	if len(out) != 10 {
		t.Fail()
	}
}

func TestPosteriorDistribution(t *testing.T) {
	//	source := rand.NewSource(time.Now().UnixNano())
	source := rand.NewSource(1)
	r := rand.New(source)

	network := initStudentNetwork()

	// test that the distribution for Node "I" given whatever is close to the
	// actual.
	NodeI := []*Node{network.Nodes[0]}
	NodeG := []*Node{network.Nodes[4]}
	NodeD := []*Node{network.Nodes[3]}

	// sort so that we can sample correctly
	network.topologicalSort()

	evidence := make(map[*Node]int)
	densities := network.PosteriorDistribution(r, NodeI, evidence)
	//	t.Log("DENSITIES 0", densities)
	probs := densities[NodeI[0]].sorted
	if math.Abs(probs[0]-NodeI[0].cpd[0].sorted[0]) > .1 {
		t.Fail()
	}

	// test that the distribution for Node "G" given various combinations of
	// "I" and "D" is close to the actual
	evidence = make(map[*Node]int)
	evidence[NodeD[0]] = 0
	evidence[NodeI[0]] = 1

	densities = network.PosteriorDistribution(
		r,
		NodeG,
		evidence)

	solution := densities[NodeG[0]].sorted
	for i, p := range densities[NodeG[0]].sorted {
		if math.Abs(p-solution[i]) > .1 {
			t.Fail()
		}
	}
}

func TestLikelihood(t *testing.T) {
	network := initStudentNetwork()
	// sort so that we can sample correctly
	network.topologicalSort()
	//	t.Log(network)

	instances := make([]map[*Node]int, 2)
	instance0 := make(map[*Node]int)
	instance0[network.Nodes[0]] = 0
	instance0[network.Nodes[1]] = 1
	instance0[network.Nodes[2]] = 1
	instance0[network.Nodes[3]] = 1
	instance0[network.Nodes[4]] = 0
	instances[0] = instance0

	instance1 := make(map[*Node]int)
	instance1[network.Nodes[0]] = 1
	instance1[network.Nodes[1]] = 1
	instance1[network.Nodes[2]] = 1
	instance1[network.Nodes[3]] = 1
	instance1[network.Nodes[4]] = 1
	instances[1] = instance1

	Likelihood := network.Likelihood(instances)
	if Likelihood[0] != .004608 {
		t.Fail()
	}
	if Likelihood[1] != .01728 {
		t.Fail()
	}
}

func TestModelLikelihood(t *testing.T) {
	network := initStudentNetwork()
	// sort so that we can sample correctly
	network.topologicalSort()

	instances := make([]map[*Node]int, 2)
	instance0 := make(map[*Node]int)
	instance0[network.Nodes[0]] = 0
	instance0[network.Nodes[1]] = 1
	instance0[network.Nodes[2]] = 1
	instance0[network.Nodes[3]] = 1
	instance0[network.Nodes[4]] = 0
	instances[0] = instance0

	instance1 := make(map[*Node]int)
	instance1[network.Nodes[0]] = 1
	instance1[network.Nodes[1]] = 1
	instance1[network.Nodes[2]] = 1
	instance1[network.Nodes[3]] = 1
	instance1[network.Nodes[4]] = 1
	instances[1] = instance1

	Likelihood := network.ModelLikelihood(instances)

	if Likelihood != -9.438166871194774 {
		t.Fail()
	}
}

func TestUpdateWeights(t *testing.T) {
	source := rand.NewSource(1)
	r := rand.New(source)

	network := NewBayesianNetwork()
	I := Node{Name: "I", cpd: []Density{NewDensity(.7, .3)}, States: 2}
	D := Node{Name: "D", cpd: []Density{NewDensity(.6, .4)}, States: 2}
	// the order of G's densities is changed from the example to reflect how
	// CPD indices are calculated in Node.go
	G := Node{Name: "G",
		cpd: []Density{
			NewDensity(.3, .4, .3),
			NewDensity(.9, .08, .02),
			NewDensity(.05, .25, .7),
			NewDensity(.5, .3, .2)}, States: 3}

	network.Nodes = []*Node{&D, &I, &G}

	network.AddEdge(&I, &G)
	network.AddEdge(&D, &G)

	// generate samples
	samples := make([]map[*Node]int, 5000)
	for i := 0; i < len(samples); i++ {
		samples[i] = network.Sample(r)
	}

	// now learn the densities for the network from the samples
	network.updateWeights(samples)

	// check that they are pretty close to the original
	if math.Abs(I.cpd[0].StateMap[0]-.7) > .05 {
		t.Fail()
	}
	if math.Abs(I.cpd[0].StateMap[1]-.3) > .05 {
		t.Fail()
	}
	if math.Abs(D.cpd[0].StateMap[0]-.6) > .05 {
		t.Fail()
	}
	if math.Abs(D.cpd[0].StateMap[1]-.4) > .05 {
		t.Fail()
	}
	// Node G
	if math.Abs(G.cpd[0].StateMap[0]-.3) > .05 {
		t.Fail()
	}
	if math.Abs(G.cpd[0].StateMap[1]-.4) > .05 {
		t.Fail()
	}
	if math.Abs(G.cpd[0].StateMap[2]-.3) > .05 {
		t.Fail()
	}

	if math.Abs(G.cpd[1].StateMap[0]-.9) > .05 {
		t.Fail()
	}
	if math.Abs(G.cpd[1].StateMap[1]-.08) > .05 {
		t.Fail()
	}
	if math.Abs(G.cpd[1].StateMap[2]-.02) > .05 {
		t.Fail()
	}

	if math.Abs(G.cpd[2].StateMap[0]-.05) > .05 {
		t.Fail()
	}
	if math.Abs(G.cpd[2].StateMap[1]-.25) > .05 {
		t.Fail()
	}
	if math.Abs(G.cpd[2].StateMap[2]-.7) > .05 {
		t.Fail()
	}

	if math.Abs(G.cpd[3].StateMap[0]-.5) > .05 {
		t.Fail()
	}
	if math.Abs(G.cpd[3].StateMap[1]-.3) > .05 {
		t.Fail()
	}
	if math.Abs(G.cpd[3].StateMap[2]-.2) > .05 {
		t.Fail()
	}

}

func TestInferNodeStates(t *testing.T) {
	network := initStudentNetwork()
	network.topologicalSort()

	// get samples
	source := rand.NewSource(1)
	samples := make([]map[*Node]int, 0)
	r := rand.New(source)
	for i := 0; i < 1000; i++ {
		samples = append(samples, network.Sample(r))
	}

	network.inferNodeStates(samples)

	min := 10
	max := 0
	for _, n := range network.Nodes {
		if n.States < min {
			min = n.States
		}
		if n.States > max {
			max = n.States
		}
	}
	if min != 2 && max != 3 {
		t.Fail()
	}
}

func TestInferBayesianNetwork(t *testing.T) {
	network := initStudentNetwork()
	network.topologicalSort()
	samples := make([]map[*Node]int, 0)

	source := rand.NewSource(1)
	r := rand.New(source)
	for i := 0; i < 1000; i++ {
		samples = append(samples, network.Sample(r))
	}

	inferred := InferBayesianNetwork(samples, 100)

	LL := inferred.ModelLikelihood(samples)

	// really we only care if the inferred network is a DAG and can be used
	// like a BN. This isn't a test of the geneticAlgorithm code
	if HasCycles(inferred) || LL >= 0 {
		t.Fail()
	}
}
