package main

import (
	"bayesiannetwork"
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
)

func loadData(fname string) [][]float64 {
	// Load a TXT file.
	f, _ := os.Open(fname)

	// Create a new reader.
	r := csv.NewReader(bufio.NewReader(f))

	data := make([][]float64, 0)

	for {
		record, err := r.Read()
		// Stop at EOF.
		if err == io.EOF {
			break
		}

		row := make([]float64, 0)
		for _, value := range record {
			v, _ := strconv.ParseFloat(value, 64)
			row = append(row, v)
		}
		data = append(data, row)
	}
	return data

}

func main() {

	data := loadData("iris.data")
	// bin the data -- bayesiannetwork requires discrete statesw
	nCols := len(data[0])
	max := make([]float64, nCols)
	min := make([]float64, nCols)

	for c := 0; c < nCols; c++ {
		min[c] = math.MaxFloat64
	}

	for _, row := range data {
		for i, v := range row {
			if max[i] < v {
				max[i] = v
			}
			if min[i] > v {
				min[i] = v
			}
		}
	}

	// discritize the features
	nBins := 3.0
	discritized := make([][]int, 0)

	for i := 0; i < len(data); i++ {
		discritized = append(discritized, make([]int, nCols))
	}
	for c := 0; c < nCols; c++ {
		for i, row := range data {
			bin := math.Floor(nBins * (row[c] - min[c]) / (max[c] - min[c]))
			bin = math.Min(nBins-1, bin)
			discritized[i][c] = int(bin)
		}
	}

	featureNames := []string{
		"sepal_length",
		"sepal_width",
		"petal_length",
		"petal_width",
		"species"}

	// make the data set into a []map[*node]int
	converted := bayesiannetwork.ConvertDataset(discritized, featureNames)

	// Typically, we would split this data set into a test and train set but
	// this is such a small data set that not all states would be observed in
	// the BN and we wouldn't be able to make inferences.
	// split the data into a training and test set
	order := rand.Perm(len(data))
	train := make([]map[*bayesiannetwork.Node]int, 0)
	test := make([]map[*bayesiannetwork.Node]int, 0)
	for i, index := range order {
		// NOTE: all instances are in the training set
		if i < 150 {
			train = append(train, converted[index])
		} else {
			test = append(test, converted[index])
		}
	}
	inferred := bayesiannetwork.InferBayesianNetwork(train, 1000)

	fmt.Println("Inferred topology:")
	for _, n := range inferred.Nodes {
		fmt.Println(n.Name)
		for _, c := range n.Children {
			fmt.Println("\t", c.Name)
		}
	}
	fmt.Println()

	mn := math.MaxFloat64
	for _, v := range inferred.Likelihood(train) {
		if v < mn {
			mn = v
		}
	}
	modelLL := inferred.ModelLikelihood(train)
	fmt.Println("Model log likelihood", modelLL)

	// get the decision matrix
	source := rand.NewSource(1)
	r := rand.New(source)

	// get the class node
	var classNode *bayesiannetwork.Node
	for n, _ := range train[0] {
		if n.Name == featureNames[len(featureNames)-1] {
			classNode = n
			break
		}
	}
	//initialize decision matrix
	confusionMatrix := make([][]int, int(nBins))
	for i := 0; i < int(nBins); i++ {
		confusionMatrix[i] = make([]int, int(nBins))
	}

	// classify the instances (first removing the class feature)
	for _, instance := range train {
		// copy the instance except the class feature
		classless := make(map[*bayesiannetwork.Node]int)
		for k, v := range instance {
			if k.Name != featureNames[len(featureNames)-1] {
				classless[k] = v
			}
		}

		dist := inferred.PosteriorDistribution(
			r,
			[]*bayesiannetwork.Node{classNode},
			classless)

		// argmax of dist / update decision matrix
		actual := instance[classNode]

		mx := 0.0
		mxState := 0
		for k, v := range dist[classNode].StateMap {
			if v > mx {
				mx = v
				mxState = k
			}
		}
		predicted := mxState

		confusionMatrix[actual][predicted] += 1
	}

	// print the confusion matrix
	fmt.Println("confusionMatrix:")
	for _, row := range confusionMatrix {
		fmt.Println(row)
	}

	// calculate the precision per class
	precision := make([]float64, int(nBins))
	for ci := 0; ci < len(confusionMatrix); ci++ {
		tp := float64(confusionMatrix[ci][ci])
		tpfp := 0.0
		for ri := 0; ri < len(confusionMatrix); ri++ {
			tpfp += float64(confusionMatrix[ci][ri])
		}
		precision[ci] = tp / tpfp
	}
	fmt.Println("precision: ", precision)
}

// Discussion:
// For a classification problem like this, it would be better to replace the InferBayesianNetwork function with a function that evaluates the model based on how well the class is predicted instead of general model likelihood.  You would also typically use a separate train and testing set.
// There are tradeoffs in using a discrete bayesian network such as this.  On the one hand, the model can approximate arbitrary distributions with multinomials with increasing numbers of buckets. On the other hand, as the number of buckets increases, the number of observations necessary to support the increased complexity grows very quickly.  A continuous bayesian network might reduce the number of necessary observations by offloading some of the intelligence into the distribution types used.
