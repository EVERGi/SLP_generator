# SLP Generator

SLP Generator is a Python-based tool designed for generating synthetic load profiles (SLPs) of electricity consumption for public buildings. Utilizing a combination of expert knowledge and machine learning models, it enables users to simulate realistic electricity consumption patterns based on building types.

## Features

- Generation of synthetic load profiles for various types of public buildings.
- Support for different building types, including sports facilities, educational institutions, cultural centers, and more.
- Integration of machine learning models for accurate profile generation.
- Easy-to-use Python class interface.

## Installation

### Prerequisites

- Python 3.6 or newer
- Git

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourgithubusername/SLP_generator.git
```

2. Navigate to the cloned repository:
```bash
cd SLP_generator
```

3. Install the required packages:
```bash
pip install .
```

## Usage

1. Import the `Generator` class from the package.
2. Initialize the `Generator` object with the path to your data.
3. Use the various methods provided by the `Generator` class to generate synthetic load profiles.

Example:
```python
from SLP_generator import Generator

# Initialize the generator
gen = Generator(data_path="path/to/your/data")

# Configure the generator
gen.configure()

# Generate a synthetic load profile
profile = gen.get_profile(scaled_cons=0.5, type="School")
print(profile)
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Evgeny Genov for creating and maintaining this project.
- Acknowledgment to the open-source community for continuous support.