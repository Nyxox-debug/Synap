import nn
import synap
import random
import struct

def load_mnist_images(filename, num_images=100):
    """Load MNIST image data"""
    with open(filename, 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))

        print("Magic number:", magic)
        print("Number of images:", num_images)
        print("Image size:", rows, "x", cols)

        images = []
        for _ in range(min(num_images, size)):
            image = []
            for _ in range(rows * cols):
                pixel = struct.unpack(">B", f.read(1))[0]
                image.append(pixel / 255.0)  # Normalize to [0, 1]
            images.append(image)
        return images, rows, cols

def load_mnist_labels(filename, num_labels=100):
    """Load MNIST label data"""
    with open(filename, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = []
        for _ in range(min(num_labels, size)):
            label = struct.unpack(">B", f.read(1))[0]
            labels.append(label)
        return labels

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

def test_mnist():
    """Train a simple neural network on MNIST digit recognition"""
    
    print("=" * 60)
    print("MNIST Handwritten Digit Recognition")
    print("=" * 60)
    print()
    
    # Load MNIST data
    print("Loading MNIST data...")
    try:
        train_images, rows, cols = load_mnist_images('train-images.idx3-ubyte', num_images=1000)
        train_labels = load_mnist_labels('train-labels.idx1-ubyte', num_labels=1000)
        test_images, _, _ = load_mnist_images('t10k-images.idx3-ubyte', num_images=100)
        test_labels = load_mnist_labels('t10k-labels.idx1-ubyte', num_labels=100)
        print(f"Loaded {len(train_images)} training images ({rows}x{cols})")
        print(f"Loaded {len(test_images)} test images")
    except FileNotFoundError:
        print("ERROR: MNIST data files not found!")
        print("Please download MNIST from: http://yann.lecun.com/exdb/mnist/")
        print("Required files:")
        print("  - train-images-idx3-ubyte")
        print("  - train-labels-idx1-ubyte")
        print("  - t10k-images-idx3-ubyte")
        print("  - t10k-labels-idx1-ubyte")
        return
    
    print()
    
    # Create model: 784 inputs (28x28) -> 128 hidden -> 10 outputs (digits 0-9)
    input_size = rows * cols  # 784
    hidden_size = 128
    output_size = 10
    
    print(f"Creating model: {input_size} -> {hidden_size} -> {output_size}")
    model = nn.MLP(input_size, [hidden_size, output_size])
    print()
    
    # Training parameters
    learning_rate = 0.01
    epochs = 5
    batch_size = 10
    
    print(f"Training parameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print()
    
    print("=" * 60)
    print("Training...")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle training data
        indices = list(range(len(train_images)))
        random.shuffle(indices)
        
        # Train in mini-batches
        for batch_start in range(0, len(train_images), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            
            # Process each example in batch
            model.zero_grad()
            batch_loss = 0.0
            
            for idx in batch_indices:
                # Get image and label
                image = train_images[idx]
                label = train_labels[idx]
                target = one_hot_encode(label)
                
                # Create tensors
                x = synap.Tensor([input_size], requires_grad=False)
                x.set_values(image)
                
                y_true = synap.Tensor([output_size], requires_grad=False)
                y_true.set_values(target)
                
                # Forward pass
                y_pred = model(x)
                
                # Loss (softmax cross-entropy)
                loss = synap.Tensor.softmax_cross_entropy(y_pred, y_true)
                
                # Backward pass (accumulate gradients)
                loss.backward()
                
                batch_loss += synap.tensor_data(loss)[0]
                
                # Check if prediction is correct
                pred_digit = predict_digit(model, image)
                if pred_digit == label:
                    correct += 1
                total += 1
            
            # Update parameters with accumulated gradients
            for param in model.parameters():
                param_values = synap.tensor_data(param)
                grad_values = param.grad_values
                
                # Average gradient over batch and update
                updated = [
                    p - learning_rate * g / len(batch_indices) 
                    for p, g in zip(param_values, grad_values)
                ]
                param.set_values(updated)
            
            epoch_loss += batch_loss
        
        # Calculate epoch statistics
        avg_loss = epoch_loss / len(train_images)
        accuracy = 100.0 * correct / total
        
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    print()
    print("=" * 60)
    print("Testing...")
    print("=" * 60)
    
    # Evaluate on test set
    correct = 0
    total = 0
    
    for i in range(len(test_images)):
        image = test_images[i]
        label = test_labels[i]
        
        pred_digit = predict_digit(model, image)
        
        if pred_digit == label:
            correct += 1
        total += 1
        
        # Show first few predictions
        if i < 10:
            status = "✓" if pred_digit == label else "✗"
            print(f"  {status} Image {i}: Predicted {pred_digit}, Actual {label}")
    
    print()
    test_accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {correct}/{total} = {test_accuracy:.2f}%")
    
    print()
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_mnist()
