import nn
import synap
import random

def create_synthetic_digit_data(num_samples=100):
    """Create synthetic digit-like data for testing without MNIST files"""
    print("Creating synthetic digit data (for testing without MNIST)...")
    
    images = []
    labels = []
    
    for _ in range(num_samples):
        # Random digit 0-9
        digit = random.randint(0, 9)
        
        # Create 28x28 = 784 pixel image
        # Each digit gets a characteristic pattern
        image = [0.0] * 784
        
        # Add some signal based on the digit
        for i in range(784):
            # Simple pattern: different digits light up different regions
            if i % 10 == digit:
                image[i] = random.uniform(0.5, 1.0)
            else:
                image[i] = random.uniform(0.0, 0.2)
        
        images.append(image)
        labels.append(digit)
    
    return images, labels

def one_hot_encode(label, num_classes=10):
    """Convert label to one-hot encoding"""
    encoding = [0.0] * num_classes
    encoding[label] = 1.0
    return encoding

def predict_digit(model, image):
    """Get predicted digit from model output"""
    x = synap.Tensor([len(image)], requires_grad=False)
    x.set_values(image)
    
    out = model(x)
    out_values = synap.tensor_data(out)
    
    # Return index of maximum value
    return out_values.index(max(out_values))

def test_digit_recognition():
    """Train a neural network on digit recognition (MNIST or synthetic data)"""
    
    print("=" * 60)
    print("Digit Recognition with Gradient Descent")
    print("=" * 60)
    print()
    
    # Try to load MNIST, fall back to synthetic data
    try:
        import struct
        
        def load_mnist_images(filename, num_images):
            with open(filename, 'rb') as f:
                magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
                images = []
                for _ in range(min(num_images, size)):
                    image = []
                    for _ in range(rows * cols):
                        pixel = struct.unpack(">B", f.read(1))[0]
                        image.append(pixel / 255.0)
                    images.append(image)
                return images
        
        def load_mnist_labels(filename, num_labels):
            with open(filename, 'rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                labels = []
                for _ in range(min(num_labels, size)):
                    label = struct.unpack(">B", f.read(1))[0]
                    labels.append(label)
                return labels
        
        print("Loading MNIST data...")
        train_images = load_mnist_images('train-images-idx3-ubyte', 500)
        train_labels = load_mnist_labels('train-labels-idx1-ubyte', 500)
        test_images = load_mnist_images('t10k-images-idx3-ubyte', 100)
        test_labels = load_mnist_labels('t10k-labels-idx1-ubyte', 100)
        print(f"✓ Loaded real MNIST data")
        print(f"  Training: {len(train_images)} images")
        print(f"  Testing: {len(test_images)} images")
        
    except (FileNotFoundError, ImportError):
        print("⚠ MNIST files not found, using synthetic data")
        train_images, train_labels = create_synthetic_digit_data(500)
        test_images, test_labels = create_synthetic_digit_data(100)
        print(f"✓ Created synthetic data")
        print(f"  Training: {len(train_images)} images")
        print(f"  Testing: {len(test_images)} images")
    
    print()
    
    # Model: 784 inputs (28x28) -> 64 hidden -> 10 outputs (digits 0-9)
    input_size = 784
    hidden_size = 64
    output_size = 10
    
    print(f"Model architecture: {input_size} -> {hidden_size} -> {output_size}")
    model = nn.MLP(input_size, [hidden_size, output_size])
    print()
    
    # Training parameters
    learning_rate = 0.05
    epochs = 10
    
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print()
    
    print("=" * 60)
    print("Training")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        
        # Train on all examples
        for i in range(len(train_images)):
            image = train_images[i]
            label = train_labels[i]
            target = one_hot_encode(label)
            
            # Create tensors
            x = synap.Tensor([input_size], requires_grad=False)
            x.set_values(image)
            
            y_true = synap.Tensor([output_size], requires_grad=False)
            y_true.set_values(target)
            
            # Forward
            y_pred = model(x)
            
            # Loss
            loss = synap.Tensor.softmax_cross_entropy(y_pred, y_true)
            
            # Backward
            model.zero_grad()
            loss.backward()
            
            # Update parameters
            for param in model.parameters():
                param_values = synap.tensor_data(param)
                grad_values = param.grad_values
                updated = [p - learning_rate * g for p, g in zip(param_values, grad_values)]
                param.set_values(updated)
            
            # Track loss and accuracy
            epoch_loss += synap.tensor_data(loss)[0]
            
            pred_digit = predict_digit(model, image)
            if pred_digit == label:
                correct += 1
        
        # Print progress
        avg_loss = epoch_loss / len(train_images)
        accuracy = 100.0 * correct / len(train_images)
        print(f"Epoch {epoch + 1:2d}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.1f}%")
    
    print()
    print("=" * 60)
    print("Testing")
    print("=" * 60)
    
    # Test
    correct = 0
    print("Sample predictions:")
    
    for i in range(len(test_images)):
        image = test_images[i]
        label = test_labels[i]
        
        pred_digit = predict_digit(model, image)
        
        if pred_digit == label:
            correct += 1
        
        # Show first 10 predictions
        if i < 10:
            status = "✓" if pred_digit == label else "✗"
            print(f"  {status} Predicted: {pred_digit}, Actual: {label}")
    
    test_accuracy = 100.0 * correct / len(test_images)
    print()
    print(f"Test Accuracy: {correct}/{len(test_images)} = {test_accuracy:.1f}%")
    print()
    print("=" * 60)

if __name__ == "__main__":
    test_digit_recognition()
