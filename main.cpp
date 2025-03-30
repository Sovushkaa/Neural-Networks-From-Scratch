#include "AdamTrainer.h"
#include "DataLoader.h"
#include "FileReader.h"
#include "FileWriter.h"
#include "NetworkBuilder.h"
#include "Trainer.h"
#include <fstream>

using namespace CNetworks;

int main() {
    auto loader = DataLoader("mnist_train.csv", "mnist_test.csv");

    NetworkBuilder b(In{784});
    b.add_layer(Out{20}, ActivationFunction::Sigmoid())
        .add_layer(Out{10}, ActivationFunction::Sigmoid());
    Network network = b.extract();

    AdamTrainer trainer =
        AdamTrainer(std::move(loader), std::move(LossFunction::Mse()),
                    Epoch_number{100}, Batch_size{5});
    network = trainer.Train(std::move(network));

    FileWriter writer = FileWriter("network_trained%txt");
    writer.write(network);
    FileReader reader = FileReader("network_trained%txt");
    auto new_loader = DataLoader("mnist_train.csv", "mnist_test.csv");
    Network new_network = reader.read<Network>();
    Trainer new_trainer =
        Trainer(std::move(new_loader), std::move(LossFunction::Mse()),
                Epoch_number{3}, Batch_size{10}, 0.01);
    new_network = new_trainer.Train(std::move(new_network));
    return 0;
}
