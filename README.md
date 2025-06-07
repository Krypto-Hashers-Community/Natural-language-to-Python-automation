# Natural Language to Python Code Converter

A Python implementation for converting natural language descriptions into Python code using fine-tuned StarCoder and CodeT5 models.

## 🚀 Features

- **Dual Model Support**: Works with both StarCoder and CodeT5 models
- **Fine-tuning Capabilities**: Customize models for your specific use cases
- **Batch Processing**: Convert multiple descriptions at once
- **Error Handling**: Robust error handling and dependency checking
- **Sample Data**: Built-in sample dataset for quick testing
- **Flexible Architecture**: Easy to extend and modify

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- At least 8GB RAM (16GB recommended for training)
- CUDA-compatible GPU (optional, for faster training)

### Dependencies
```
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.20.0
```

## 🛠️ Installation

### Option 1: Automatic Installation
```bash
git clone https://github.com/Krypto-Hashers-Community/Natural-language-to-Python-automation
cd Natural-language-to-Python-automation
pip install -r requirements.txt
```

### Option 2: Manual Installation

1. **Install PyTorch**
   ```bash
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Install other dependencies**
   ```bash
   pip install transformers datasets accelerate
   ```

### Option 3: Using Conda
```bash
# Create a new environment
conda create -n nltocode python=3.9
conda activate nltocode

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
conda install transformers datasets -c huggingface
pip install accelerate
```

## 🚀 Quick Start

### Basic Usage

```python
from nl_to_code_converter import NLToCodeConverter

# Initialize the converter
converter = NLToCodeConverter("starcoder")  # or "codet5"

# Convert natural language to Python code
code = converter.convert("Create a function that adds two numbers")
print(code)
```

### Fine-tuning Example

```python
# Prepare your training data
training_data = [
    {
        "natural_language": "Create a function that adds two numbers",
        "python_code": "def add_numbers(a, b):\n    return a + b"
    },
    # Add more examples...
]

# Fine-tune the model
converter = NLToCodeConverter("starcoder")
converter.train(training_data, output_dir="./my_finetuned_model", epochs=3)

# Use the fine-tuned model
code = converter.convert("Write a function to multiply two numbers")
```

### Batch Processing

```python
descriptions = [
    "Create a list of even numbers from 0 to 20",
    "Write a function to check if a number is prime",
    "Create a dictionary with fruit names and colors"
]

codes = converter.batch_convert(descriptions)
for desc, code in zip(descriptions, codes):
    print(f"Input: {desc}")
    print(f"Output: {code}")
    print("-" * 50)
```

## 📊 Model Comparison

| Feature | StarCoder | CodeT5 |
|---------|-----------|--------|
| **Architecture** | Decoder-only (GPT-style) | Encoder-Decoder (T5-style) |
| **Training** | Causal Language Modeling | Sequence-to-Sequence |
| **Best For** | Code completion, generation | Code translation, summarization |
| **Memory Usage** | Higher | Lower |
| **Speed** | Slower | Faster |

## 🔧 Configuration

### Model Parameters

```python
# StarCoder Configuration
starcoder_converter = NLToCodeConverter("starcoder")
starcoder_converter.fine_tuner.model_name = "bigcode/starcoder"  # or "bigcode/starcoderbase"

# CodeT5 Configuration  
codet5_converter = NLToCodeConverter("codet5")
codet5_converter.fine_tuner.model_name = "Salesforce/codet5-base"  # or "Salesforce/codet5-large"
```

### Training Parameters

```python
converter.train(
    training_data=your_data,
    output_dir="./custom_model",
    epochs=5,                    # Number of training epochs
    batch_size=4,               # Batch size for training
    learning_rate=5e-5,         # Learning rate
    max_length=512              # Maximum sequence length
)
```

## 📁 Data Format

### Training Data Structure
```json
[
    {
        "natural_language": "Create a function that calculates factorial",
        "python_code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    },
    {
        "natural_language": "Write a loop to print numbers 1 to 10",
        "python_code": "for i in range(1, 11):\n    print(i)"
    }
]
```

### Loading Custom Data
```python
# From JSON file
converter.load_training_data("path/to/your/data.json")

# From Python list
custom_data = [
    {"natural_language": "...", "python_code": "..."},
    # more examples
]
converter.train(custom_data)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. PyTorch Installation Error
```
ModuleNotFoundError: No module named 'torch._C'
```
**Solution:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size: `per_device_train_batch_size=1`
- Use gradient accumulation: `gradient_accumulation_steps=8`
- Use CPU training: Set `device_map=None`

#### 3. Model Download Issues
```
HTTPError: 403 Client Error
```
**Solutions:**
- Check internet connection
- Try using a VPN
- Use local model files

### Dependency Check
```python
from nl_to_code_converter import check_dependencies
check_dependencies()
```

## 📈 Performance Tips

### For Better Code Generation:
1. **Use specific, clear descriptions**
   ```python
   # Good
   "Create a function that takes a list of numbers and returns the sum"
   
   # Less specific
   "Make a function for adding"
   ```

2. **Include context in your training data**
   ```python
   # Include various coding patterns
   training_data = [
       {"natural_language": "Create a class with constructor", "python_code": "class MyClass:\n    def __init__(self, value):\n        self.value = value"},
       {"natural_language": "Write error handling code", "python_code": "try:\n    # code here\n    pass\nexcept Exception as e:\n    print(f'Error: {e}')"}
   ]
   ```

3. **Fine-tune on domain-specific data**
   - Web scraping code examples
   - Data science snippets
   - API integration patterns

### For Faster Training:
- Use smaller models for prototyping
- Implement early stopping
- Use mixed precision training
- Enable gradient checkpointing

## 🧪 Testing

### Run Basic Tests
```python
python -c "from nl_to_code_converter import check_dependencies; check_dependencies()"
```

### Test Code Generation
```python
# Test with sample data
converter = NLToCodeConverter("starcoder")
sample_data = converter.create_sample_data()
print(f"Sample data loaded: {len(sample_data)} examples")

# Test conversion
result = converter.convert("Create a simple calculator function")
print(f"Generated code: {result}")
```

## 📚 Examples

### Example 1: Data Processing
```python
# Input
description = "Create a function that filters even numbers from a list"

# Output
def filter_even_numbers(numbers):
    return [num for num in numbers if num % 2 == 0]
```

### Example 2: File Operations
```python
# Input
description = "Write code to read a CSV file and print the first 5 rows"

# Output
import pandas as pd
df = pd.read_csv('file.csv')
print(df.head())
```

### Example 3: API Integration
```python
# Input
description = "Create a function to make a GET request to an API"

# Output
import requests

def make_api_request(url):
    response = requests.get(url)
    return response.json()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone <repository-url>
cd natural-language-to-python-converter
pip install -e .
pip install -r requirements-dev.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [StarCoder](https://github.com/bigcode-project/starcoder) by BigCode
- [CodeT5](https://github.com/salesforce/CodeT5) by Salesforce
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Krypto-Hashers-Community/Natural-language-to-Python-automation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Krypto-Hashers-Community/Natural-language-to-Python-automation/discussions)
- **Email**: bhowmicksaurav28@gmail.com

## 🔮 Roadmap

- [ ] Support for more programming languages (JavaScript, Java, C++)
- [ ] Web interface for easier usage
- [ ] Integration with popular IDEs
- [ ] Model quantization for mobile deployment
- [ ] Real-time code suggestion API
- [ ] Code explanation and documentation generation

---

**Made with ❤️ for the developer community**
