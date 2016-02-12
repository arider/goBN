package bayesiannetwork

import (
	"math"
	"sort"
)

// add an index field for sorting slices
type Slice struct {
	sort.Interface
	index []int
}

func (s Slice) Swap(i, j int) {
	s.Interface.Swap(i, j)
	s.index[i], s.index[j] = s.index[j], s.index[i]
}

func NewSlice(n sort.Interface) *Slice {
	s := &Slice{Interface: n, index: make([]int, n.Len())}
	for i := range s.index {
		s.index[i] = i
	}
	return s
}

func NewIntIndexSlice(n ...int) *Slice         { return NewSlice(sort.IntSlice(n)) }
func NewFloat64IndexSlice(n ...float64) *Slice { return NewSlice(sort.Float64Slice(n)) }
func NewStringIndexSlice(n ...string) *Slice   { return NewSlice(sort.StringSlice(n)) }

func HasCycles(net *BayesianNetwork) (hasCycles bool) {
	// gone is Nodes that we are pretending are no longer in the network
	nodes := net.Nodes
	gone := make(map[*Node]bool)
	remaining := make(map[*Node]bool)
	for _, n := range nodes {
		remaining[n] = true
	}

	for n, _ := range remaining {
		visited := make(map[*Node]bool)
		hasCycles := visitDFS(n, visited, gone)
		if hasCycles {
			return true
		}
	}
	return false
}

// depth first search to detect cycles
func visitDFS(n *Node, visited map[*Node]bool, gone map[*Node]bool) (hasCycle bool) {
	//	fmt.Println("visited check", n.Name, visited[n])
	detected := false
	if _, exists := visited[n]; exists {
		return true
	} else {
		visited[n] = true
		for _, child := range n.Children {

			if _, exists := gone[n]; !exists {
				if visitDFS(child, visited, gone) {
					detected = true
				}
			}
		}
	}
	return detected
}

// convert a network's topology into a slice of binary integers
func networkToBinary(nodes []*Node) []int {
	bits := make([]int, len(nodes)*len(nodes))
	NodeIndex := make(map[*Node]int)

	for i, Node := range nodes {
		NodeIndex[Node] = i
	}
	for i, Node := range nodes {
		//		fmt.Print(Node.Name, "->")
		for _, child := range Node.Children {
			bits[i*len(nodes)+NodeIndex[child]] = 1
			//			fmt.Print(child.Name, " ")
		}
		//		fmt.Println()
	}

	return bits
}

// convert a bitset into a network topology, update the topology in the given
// Nodes. NOTE: Need to be careful with this since the order of Nodes in the
// network is changed by topologicalSort.
func (network *BayesianNetwork) binaryToTopology(NodeOrder []*Node, bits []int) {
	nNodes := int(math.Sqrt(float64(len(bits))))

	// clear the existing topology
	for _, n := range network.Nodes {
		n.Children = make([]*Node, 0)
		n.Parents = make([]*Node, 0)
		n.cpd = make([]Density, 0)
	}

	// add the edges
	var n *Node
	for i := 0; i < len(bits); i++ {
		if i%nNodes == 0 {
			n = NodeOrder[i/nNodes]
		}
		if bits[i] == 1 {
			childNode := NodeOrder[i%nNodes]
			n.Children = append(n.Children, childNode)
			childNode.Parents = append(childNode.Parents, n)
		}
	}
}

// Function to take a matrix (discritized) and make a []map[*Node]int
func ConvertDataset(data [][]int, featureNames []string) []map[*Node]int {
	// convert data sets into []map[*bayesiannetwork.Node]int for BayesianNetwork
	nodes := make([]*Node, 0)
	for _, n := range featureNames {
		nodes = append(nodes, &Node{Name: n})
	}

	converted := make([]map[*Node]int, 0)
	for _, row := range data {
		c := make(map[*Node]int)
		for index, el := range row {
			c[nodes[index]] = el
		}
		converted = append(converted, c)
	}

	return converted
}
