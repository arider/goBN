package geneticalgorithm

import (
	"math/rand"
)

// randomly perturb the current best and update it when it reduces the score
func Evolve(
	scoreFunction func([]int) float64,
	mutateCheckFunction func(bits []int) bool,
	genomeSize int,
	iterations int,
	minimize bool) []int {

	//	check := make([]int, genomeSize)
	best := mutate(make([]int, genomeSize), mutateCheckFunction)
	value := scoreFunction(best)
	bestScore := value

	for i := 0; i < iterations; i++ {
		child := mutate(best, mutateCheckFunction)
		score := scoreFunction(child)
		if minimize && score < bestScore {
			bestScore = score
			best = child
		} else if !minimize && score > bestScore {
			bestScore = score
			best = child
		}
	}

	return best
}

// randomly perturb the individual a random amount
// TODO: add some options here
// takes a mutateCheckFunction that returns whether the
// proposed bit set is valid
func mutate(individual []int,
	mutateCheckFunction func(bits []int) bool) []int {

	candidate := make([]int, len(individual))
	// select a random number of random indices
	nChanges := rand.Intn(len(individual))
	order := rand.Perm(len(individual))[:nChanges]

	for _, index := range order {
		check := make([]int, len(individual))
		copy(check, candidate)
		if check[index] == 0 {
			check[index] = 1
		} else {
			check[index] = 0
		}
		if mutateCheckFunction(check) {
			candidate = check
		}
	}

	return candidate
}
