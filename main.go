package main

import (
	"github.com/NOX73/go-neural"
	"github.com/NOX73/go-neural/learn"
	"github.com/NOX73/go-neural/persist"
	"os"
	"log"
	"bufio"
	"os/signal"
	"bytes"
	"io/ioutil"
	"math/rand"
	"time"
	"encoding/csv"
	"strings"
	"strconv"
	"github.com/cheggaaa/pb"
	"fmt"
)

const (
	jsonFile   = "engine.json"
//	csvLearn = "csv/11-01.csv"
	csvLearn = "csv/cat.csv"
	csvTest = "csv/11-02.csv"
)

func main() {
	rand.Seed(time.Now().Unix())

	createLangNetwork()

	n := loadNetwork()

	testEngine(n)

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)

learnLoop:
	for !checkEngine(n) {

		learnEngine(n)

		select {
		case <-c:
			log.Println("Interrupt !")
			break learnLoop
		default:
		}

	}
	testEngine(n)

	saveNetwork(n)
}

var (
	learningSpeed = 0.2
)

func checkEngine(n *neural.Network) bool {
	var out []float64

	code, sample := getRandomSample(csvTest)
	out = n.Calculate(sample);
	log.Println("Check", code, getOutCode(out))
	time.Sleep(5 * time.Second)

	/*
	if out[0] < 0.9 {
		return false
	}

	return true
	*/

	return false
}

func getRandomSample(path string) (string, []float64) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}

	r := bufio.NewReader(file)
	data, err := ioutil.ReadAll(r)
	if err != nil {
		log.Fatal(err)
	}

	lines := bytes.Split(data, []byte("\n"))

	for {
		line := string(lines[rand.Intn(len(lines))])

		rc := csv.NewReader(strings.NewReader(line))
		rc.Comma = ';'

		values, err := rc.ReadAll()
		if err != nil {
			log.Fatal(err)
		}

		if len(values) == 0 {
			fmt.Println(values)
		} else {
			return getSampleFromValues(values[0]);
		}
	}
}

func getSampleFromValues(values []string) (string, []float64) {
	var result []float64
	for y := 0; y < 32; y++ {
		for x := 0; x < 128; x++ {
			value, err := strconv.ParseFloat(values[y * 128 + x + 1], 64)
			if err != nil {
				log.Fatal(err)
			}
			result = append(result, value / 256)
		}
	}
	return values[0], result
}

func testEngine(n *neural.Network) {
	_, out := getRandomSample(csvTest)
	log.Println(csvTest, getOutCode(n.Calculate(out)))
}

func learnEngine(n *neural.Network) {

	count := 1000
	bar := pb.StartNew(count)

	for i := 0; i < count; i++ {
		bar.Increment()
		code, sample := getRandomSample(csvLearn)
		learn.Learn(n, sample, getCodeOut(code), learningSpeed)
	}
	bar.Finish()
}

func getCodeOut(code string) []float64 {
	var result [60]float64
	value, _ := strconv.ParseInt(code, 10,64)
	v := fmt.Sprintf("%06d", value);
	for i := 0; i < 6; i++ {
		el, _ := strconv.ParseInt(string([]rune(v)[i]), 10, 64)
		result[i*10 + int(el)] = 1
	}
	return result[:]
}

func getOutCode(out []float64) string {
	values := []interface{}{}
	for i := 0; i < 6; i++ {
		slice := out[i * 10:(i+1)*10]
		maxVal := slice[0]
		maxIndex := 0
		for j:=1; j<10; j++ {
			if slice[j] > maxVal {
				maxVal = slice[j]
				maxIndex = j
			}
		}
		values = append(values, maxIndex, maxVal)
	}
	return fmt.Sprint(values)
}

func createLangNetwork() {
	if _, err := os.Stat(jsonFile); err == nil {
		return
	}
	n := neural.NewNetwork(128*32, []int{500, 60})
	n.RandomizeSynapses()

	persist.ToFile(jsonFile, n)
}

func loadNetwork() *neural.Network {
	return persist.FromFile(jsonFile)
}

func saveNetwork(n *neural.Network) {
	persist.ToFile(jsonFile, n)
}

