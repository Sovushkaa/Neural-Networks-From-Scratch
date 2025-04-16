## Нейросети с нуля на C++

В данном репозитории представлена реализация полносвязных нейронных сетей на языке С++. Поддерживаются следующие
алгоритмы обучения:

- GD (gradient descent)
- SGD (stochastic gradient descent)
- SGD with momentum
- ADAM

Для начала работы достаточно подключить хедер:

```cpp
#include "DenseNetwork.h"
```

## Датасет

Тестирование происходило на стандартном датасете MNIST. Вот так можно его загрузить:

```cpp
auto loader = DataLoader("mnist_train.csv", "mnist_test.csv");
```

## Построение нейронной сети

```cpp
NetworkBuilder b(In{784});
b.add_layer(Out{20}, ActivationFunction::Sigmoid())
.add_layer(Out{10}, ActivationFunction::Sigmoid());
Network network = b.extract();
```

## Обучение с различными алгоритмами:

GD:

```cpp
GDTrainer trainer =
GDTrainer(std::move(loader), std::move(LossFunction::Mse()),
Epoch_number{5}, 0.01); //learning rate
network = trainer.Train(std::move(network));
```

SGD:

```cpp
SGDTrainer trainer =
SGDTrainer(std::move(loader), std::move(LossFunction::Mse()),
Epoch_number{5}, Batch_size{3}, 0.001); //learning rate
network = trainer.Train(std::move(network));
```

SGD with momentum:

```cpp
SGDWithMomentumTrainer trainer =
SGDWithMomentumTrainer(std::move(loader), std::move(LossFunction::Mse()),
Epoch_number{5}, Batch_size{3}, 0.0001, 0.9); //learning rate, momentum_term
network = trainer.Train(std::move(network));
```

ADAM:

```cpp
AdamTrainer trainer =
AdamTrainer(std::move(loader), std::move(LossFunction::Mse()),
Epoch_number{5}, Batch_size{3}, 0.001, 0.9, 0.999, 1e-8); // alpha, beta1, beta2, eps
network = trainer.Train(std::move(network));
```

## Запись нейронной сети в файл:

```cpp
FileWriter writer = FileWriter("network.txt");
writer.write(network);
```

## И чтение её оттуда:

```cpp
FileReader reader = FileReader("network.txt");
Network new_network = reader.read<Network>();
```

## Анализ

В репозитории представлен файл NetworkAnalysis.ipynb, в котором собрана небольшая аналитика работы библиотеки.
Экспериментально выяснилось, что в ней SGD обучается быстрее при маленьких размерах батча.
Было проверено, что аналогичное верно и в Python, но в Python при маленьких размерах батча страдает скорость, здесь же
она остается высокой.

