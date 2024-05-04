package embedding

import (
	"context"
	"log"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sashabaranov/go-openai"
)

type ContentStore[T any] interface {
	GetContent(item T) string
}

type EmbeddingStore[T any] interface {
	StoreEmbedding(item T, embedding []float32)
}

// Combine both interfaces if an object needs to provide both functionalities
type EmbeddingAdapter[T any] interface {
	ContentStore[T]
	EmbeddingStore[T]
}

// EmbeddingGenerator defines the interface for generating embeddings.
type EmbeddingGenerator interface {
	GenerateEmbedding(context context.Context, content string) ([]float32, error)
}

// RateLimiter defines the interface for adjusting worker settings based on performance.
type RateLimiter interface {
	AdjustConcurrency(elapsed time.Duration) int
	Concurrency() int
	RequestLimit() int
	Period() time.Duration
}

// EmbeddingService structures the embedding generation process using dependency injection.
type EmbeddingService[T any] struct {
	Generator EmbeddingGenerator
	Adapter   EmbeddingAdapter[T]
	Limiter   RateLimiter
}

// NewEmbeddingService creates a new instance of EmbeddingService.
func NewEmbeddingService[T any](generator EmbeddingGenerator, adapter EmbeddingAdapter[T], limiter RateLimiter) *EmbeddingService[T] {
	return &EmbeddingService[T]{
		Generator: generator,
		Adapter:   adapter,
		Limiter:   limiter,
	}
}

// Generate processes all verses and handles worker adjustments.
func (es *EmbeddingService[T]) GenerateEmbeddings(ctx context.Context, items []T) error {
	var total int64
	wg := &sync.WaitGroup{}
	for i := 0; i < len(items); i += es.Limiter.RequestLimit() {
		var count int64

		totalItems := len(items)
		numWorkers := es.Limiter.Concurrency()
		batchSize := min(es.Limiter.RequestLimit(), totalItems-i)
		chunkSize := batchSize / numWorkers

		wg.Add(numWorkers)

		startTime := time.Now()
		for j := 0; j < numWorkers; j++ {

			start := i + j*chunkSize
			end := start + chunkSize

			// Adjust the end for the last worker or if end exceeds total items
			if end > totalItems {
				end = totalItems
			} else if j == numWorkers-1 && end < i+batchSize { // adjust when there was integer truncation on the chunkSize calculation
				end = min(totalItems, i+batchSize)
			}

			go func(start, end int) {
				defer wg.Done()
				for _, item := range items[start:end] {
					content := es.Adapter.GetContent(item)
					embedding, err := es.Generator.GenerateEmbedding(ctx, content)
					if err != nil {
						log.Printf("Error generating embeddings: %v '%v'", err, item)
						continue
					}
					es.Adapter.StoreEmbedding(item, embedding)
					atomic.AddInt64(&count, 1)
				}
			}(start, end)
		}
		wg.Wait()
		total += count
		elapsed := time.Since(startTime)
		log.Printf("Processed %d verses in %s with %d workers. Total processed %d\n", count, elapsed, numWorkers, total)
		es.Limiter.AdjustConcurrency(elapsed)
	}
	return nil
}

type EmbeddingClient interface {
	CreateEmbeddings(ctx context.Context, req openai.EmbeddingRequestConverter) (openai.EmbeddingResponse, error)
}

// OpenAIGenerator is an implementation of the EmbeddingGenerator interface.
type OpenAIGenerator struct {
	Client     EmbeddingClient
	Model      openai.EmbeddingModel
	Dimensions int
}

func NewOpenAIGenerator(client EmbeddingClient, model openai.EmbeddingModel, dimensions int) *OpenAIGenerator {
	return &OpenAIGenerator{
		Client:     client,
		Model:      model,
		Dimensions: dimensions,
	}
}

func (sg *OpenAIGenerator) GenerateEmbedding(context context.Context, content string) ([]float32, error) {
	embeddingRequest := openai.EmbeddingRequest{
		Input:      []string{content},
		Model:      sg.Model,
		Dimensions: sg.Dimensions,
	}
	response, err := sg.Client.CreateEmbeddings(context, embeddingRequest)
	if err != nil {
		return nil, err
	}
	return response.Data[0].Embedding, nil
}

// SteadyRateLimiter is an implementation of the WorkerAdjuster interface.
type SteadyRateLimiter struct {
	concurrency  int
	period       time.Duration
	requestLimit int
}

func NewSteadyRateLimiter(requestLimit int, period time.Duration, numWorkers int) *SteadyRateLimiter {
	return &SteadyRateLimiter{
		concurrency:  numWorkers,
		period:       period,
		requestLimit: requestLimit,
	}
}

func (sa *SteadyRateLimiter) AdjustConcurrency(elapsed time.Duration) int {
	ratio := float64(elapsed) / float64(sa.period)
	if elapsed > sa.period {
		sa.concurrency = int(math.Ceil(float64(sa.concurrency) * ratio))
	} else if elapsed < sa.period {
		sa.concurrency = max(1, int(float64(sa.concurrency)*ratio)+1)
	}
	return sa.concurrency
}

func (sa *SteadyRateLimiter) Concurrency() int {
	return sa.concurrency
}

func (sa *SteadyRateLimiter) RequestLimit() int {
	return sa.requestLimit
}

func (sa *SteadyRateLimiter) Period() time.Duration {
	return sa.period
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
