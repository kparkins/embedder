package embedder

import (
	"context"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// Mock for EmbeddingGenerator
type MockGenerator struct {
	mock.Mock
}

func (m *MockGenerator) GenerateEmbedding(ctx context.Context, content any) ([]float32, error) {
	args := m.Called(ctx, content)
	return args.Get(0).([]float32), args.Error(1)
}

// Mock for EmbeddingAdapter
type MockAdapter[T any] struct {
	mock.Mock
}

func (m *MockAdapter[T]) GetContent(item T) any {
	args := m.Called(item)
	return args.Get(0)
}

func (m *MockAdapter[T]) StoreEmbedding(item T, embedding []float32) {
	m.Called(item, embedding)
}

type MockEmbeddingClient struct {
	mock.Mock
}

func (m *MockEmbeddingClient) CreateEmbeddings(ctx context.Context, req openai.EmbeddingRequestConverter) (openai.EmbeddingResponse, error) {
	args := m.Called(ctx, req)
	return args.Get(0).(openai.EmbeddingResponse), args.Error(1)
}

// Testing OpenAIGenerator.GenerateEmbedding
func TestOpenAIGenerator_GenerateEmbedding(t *testing.T) {
	mockClient := new(MockEmbeddingClient)
	gen := NewOpenAIGenerator(mockClient, openai.SmallEmbedding3, 128)

	ctx := context.Background()
	content := "test content"
	expectedEmbedding := []float32{0.1, 0.2, 0.3}
	mockResponse := openai.EmbeddingResponse{
		Data: []openai.Embedding{{Embedding: expectedEmbedding}},
	}

	// Setup the expected call
	mockClient.On("CreateEmbeddings", ctx, mock.AnythingOfType("openai.EmbeddingRequest")).Return(mockResponse, nil)

	embedding, err := gen.GenerateEmbedding(ctx, content)
	assert.NoError(t, err)
	assert.Equal(t, expectedEmbedding, embedding)

	mockClient.AssertExpectations(t)
}

// Testing SteadyRateLimiter
func TestSteadyRateLimiter(t *testing.T) {
	limiter := NewSteadyRateLimiter(100, time.Minute, 10)

	assert.Equal(t, 10, limiter.Concurrency())
	assert.Equal(t, 100, limiter.RequestLimit())
	assert.Equal(t, time.Minute, limiter.Period())

	// Test AdjustConcurrency
	limiter.AdjustConcurrency(30 * time.Second) // Less than period
	assert.Equal(t, 6, limiter.Concurrency())

	limiter.AdjustConcurrency(90 * time.Second) // More than period
	assert.Equal(t, 9, limiter.Concurrency())

	limiter.AdjustConcurrency(0 * time.Second) // extreme case
	assert.Equal(t, 1, limiter.Concurrency())
}

// Testing EmbeddingService.GenerateEmbeddings
func TestEmbeddingService_GenerateEmbeddings(t *testing.T) {
	mockGen := &MockGenerator{}
	mockAdapter := &MockAdapter[int]{}
	mockLimiter := &SteadyRateLimiter{
		concurrency:  1,
		requestLimit: 1,
		period:       time.Minute,
	}

	service := NewEmbeddingService[int](mockGen, mockAdapter, mockLimiter)

	mockGen.On("GenerateEmbedding", mock.Anything, mock.Anything).Return([]float32{0.1, 0.2, 0.3}, nil)
	mockAdapter.On("GetContent", mock.Anything).Return("sample content")
	mockAdapter.On("StoreEmbedding", mock.Anything, mock.Anything).Return()

	items := []int{1, 2, 3}
	err := service.GenerateEmbeddings(context.TODO(), items)
	assert.NoError(t, err)

	mockGen.AssertExpectations(t)
	mockAdapter.AssertExpectations(t)
}
func TestSteadyRateLimiter_AdjustConcurrency(t *testing.T) {
	tests := []struct {
		name            string
		initialWorkers  int
		elapsedTime     time.Duration
		expectedWorkers int
	}{
		{
			name:            "Increase Concurrency",
			initialWorkers:  10,
			elapsedTime:     90 * time.Second, // More than period
			expectedWorkers: 15,               // Expected increase
		},
		{
			name:            "Decrease Concurrency",
			initialWorkers:  10,
			elapsedTime:     30 * time.Second, // Less than period
			expectedWorkers: 6,                // Expected decrease
		},
		{
			name:            "No Change",
			initialWorkers:  10,
			elapsedTime:     60 * time.Second, // Exactly the period
			expectedWorkers: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			limiter := NewSteadyRateLimiter(100, 60*time.Second, tt.initialWorkers)
			newConcurrency := limiter.AdjustConcurrency(tt.elapsedTime)
			assert.Equal(t, tt.expectedWorkers, newConcurrency)
		})
	}
}
func TestCalculateBatchIndexes(t *testing.T) {
	tests := []struct {
		name          string
		index         int
		worker        int
		totalItems    int
		numWorkers    int
		requestLimit  int
		expectedStart int
		expectedEnd   int
	}{
		{
			name:          "Even Distribution",
			index:         0,
			worker:        0,
			totalItems:    100,
			numWorkers:    10,
			requestLimit:  100,
			expectedStart: 0,
			expectedEnd:   10,
		},
		{
			name:          "Last Worker Excess",
			index:         0,
			worker:        9,
			totalItems:    95,
			numWorkers:    10,
			requestLimit:  100,
			expectedStart: 90,
			expectedEnd:   95,
		},
		{
			name:          "Uneven Distribution",
			index:         50,
			worker:        0,
			totalItems:    120,
			numWorkers:    4,
			requestLimit:  40,
			expectedStart: 50,
			expectedEnd:   60,
		},
		{
			name:          "Single Worker",
			index:         0,
			worker:        0,
			totalItems:    10,
			numWorkers:    1,
			requestLimit:  100,
			expectedStart: 0,
			expectedEnd:   10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			limiter := &SteadyRateLimiter{
				concurrency:  tt.numWorkers,
				requestLimit: tt.requestLimit,
			}
			service := NewEmbeddingService[int](nil, nil, limiter)
			start, end := service.calculateBatchIndexes(tt.index, tt.worker, tt.totalItems)
			assert.Equal(t, tt.expectedStart, start)
			assert.Equal(t, tt.expectedEnd, end)
		})
	}
}
