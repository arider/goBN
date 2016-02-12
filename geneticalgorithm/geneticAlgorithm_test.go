package geneticalgorithm

import "testing"
import "math/rand"

func scoreFunction(b []int) float64 {
	solution := []int{0, 1, 1, 1, 0}
	count := 0.0
	for i, _ := range b {
		if solution[i] != b[i] {
			count++
		}
	}
	return count
}
func mutateCheckFunction([]int) bool { return true }

func TestMutate(t *testing.T) {
	rand.Seed(1)
	mutate(make([]int, 5), mutateCheckFunction)
}

func TestEvolve(t *testing.T) {
	rand.Seed(1)
	solution := []int{0, 1, 1, 1, 0}
	best := Evolve(scoreFunction, mutateCheckFunction, 5, 100, true)
	for i, b := range best {
		if solution[i] != b {
			t.Fail()
		}
	}
}
