#pragma once
#include "Except.h"
#include "NetworkBuilder.h"
#include "SGDTrainer.h"
#include "GDTrainer.h"
#include "SGDWithMomentumTrainer.h"
#include "AdamTrainer.h"
#include "FileWriter.h"
#include "FileReader.h"
#include <fstream>

namespace CNetworks {
    void TestGD() {
        std::cout << "TestGD" << std::endl;
        auto loader = DataLoader("mnist_train.csv", "mnist_test.csv");
        NetworkBuilder b(In{784});
        b.add_layer(Out{20}, ActivationFunction::Sigmoid())
                .add_layer(Out{10}, ActivationFunction::Sigmoid());
        Network network = b.extract();

        GDTrainer trainer =
                GDTrainer(std::move(loader), std::move(LossFunction::Mse()),
                          Epoch_number{5}, 0.0001);
        network = trainer.Train(std::move(network));
        std::cout << "Test successfully passed!" << std::endl << std::endl;
    }

    void TestSGD() {
        std::cout << "TestSGD" << std::endl;
        auto loader = DataLoader("mnist_train.csv", "mnist_test.csv");
        NetworkBuilder b(In{784});
        b.add_layer(Out{20}, ActivationFunction::Sigmoid())
                .add_layer(Out{10}, ActivationFunction::Sigmoid());
        Network network = b.extract();

        SGDTrainer trainer =
                SGDTrainer(std::move(loader), std::move(LossFunction::Mse()),
                           Epoch_number{5}, Batch_size{3}, 0.001);
        network = trainer.Train(std::move(network));
        std::cout << "Test successfully passed!" << std::endl << std::endl;
    }

    void TestSGDWithMomentum() {
        std::cout << "TestSGDWithMomentum" << std::endl;
        auto loader = DataLoader("mnist_train.csv", "mnist_test.csv");
        NetworkBuilder b(In{784});
        b.add_layer(Out{20}, ActivationFunction::Sigmoid())
                .add_layer(Out{10}, ActivationFunction::Sigmoid());
        Network network = b.extract();

        SGDWithMomentumTrainer trainer =
                SGDWithMomentumTrainer(std::move(loader), std::move(LossFunction::Mse()),
                                       Epoch_number{5}, Batch_size{3}, 0.0001, 0.9);
        network = trainer.Train(std::move(network));
        std::cout << "Test successfully passed!" << std::endl << std::endl;
    }

    void TestAdam() {
        std::cout << "TestAdam" << std::endl;
        auto loader = DataLoader("mnist_train.csv", "mnist_test.csv");
        NetworkBuilder b(In{784});
        b.add_layer(Out{20}, ActivationFunction::Sigmoid())
                .add_layer(Out{10}, ActivationFunction::Sigmoid());
        Network network = b.extract();

        AdamTrainer trainer =
                AdamTrainer(std::move(loader), std::move(LossFunction::Mse()),
                            Epoch_number{5}, Batch_size{3}, 0.001, 0.9, 0.999, 1e-8);
        network = trainer.Train(std::move(network));
        std::cout << "Test successfully passed!" << std::endl << std::endl;
    }

    void TestFiles() {
        std::cout << "TestFiles" << std::endl;
        auto loader = DataLoader("mnist_train.csv", "mnist_test.csv");
        NetworkBuilder b(In{784});
        b.add_layer(Out{20}, ActivationFunction::Sigmoid())
                .add_layer(Out{10}, ActivationFunction::Sigmoid());
        Network network = b.extract();
        FileWriter writer = FileWriter("random_network.txt");
        writer.write(network);
        FileReader reader = FileReader("random_network.txt");
        auto new_loader = DataLoader("mnist_train.csv", "mnist_test.csv");
        Network new_network = reader.read<Network>();
        std::cout << "Test successfully passed!" << std::endl << std::endl;
    }


    void RunALLTests() {
        TestGD();
        TestSGD();
        TestSGDWithMomentum();
        TestAdam();
        TestFiles();
    }
}
