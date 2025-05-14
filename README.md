# Java Convolutional Neural Network
_A CNN built from scratch in Java to recognize handwritten digits._

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview
This project is a Convolutional Neural Network (CNN) implemented from scratch in Java. It is currently designed to process grayscale images (like the MNIST dataset) and classify them into digits (0-9).

---

## Features
- Developed a fully connected nueral network from scratch in Java, supporting multiple hideen layers and customizable architecture.
- Added a drawing pad to test network predictions.
- Implemented a custom Matrix class to handle all linear algebra operations, including dot products, element-wise functions, and matrix transformations.
- Built forward propogation and backpropogation logic to train networks using gradient descent and cost functions.
- Enabled training of networks on labeled data, such as MNIST and Google’s Quick Draw dataset, with adjustable learning rate, epochs, and batch sizes.
- Added functionality to save and load trained models, allowing quick reuse.
- Designed the system to be modular and extendable for future additions.

---

## Installation
1. Close the repository
  ```bash
  git clone https://github.com/Brae2357/Convolutional-Neural-Network.git
  cd Convolutional-Neural-Network
  ```
2. Open the project in IntelliJ.

---

## Future Improvements
- Working towards optimized convolutional operations.
- Expanding network to handle RGB images.
- Enable GPU support for faster training.

---

## License
This project is open-source under the MIT License.
