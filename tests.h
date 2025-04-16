#pragma once
#include "DenseNetwork.h"

namespace CNetworks {
    inline void TestGD() {
        std::cout << "TestGD" << std::endl;
        auto loader = DataLoader("mnist_train.csv", "mnist_test.csv");
        NetworkBuilder b(In{784});
        b.add_layer(Out{20}, ActivationFunction::Sigmoid())
                .add_layer(Out{10}, ActivationFunction::Sigmoid());
        Network network = b.extract();

        GDTrainer trainer =
                GDTrainer(std::move(loader), std::move(LossFunction::Mse()),
                          Epoch_number{5}, 0.01);
        network = trainer.Train(std::move(network));
        std::cout << "Test successfully passed!" << std::endl << std::endl;
    }

    inline void TestSGD() {
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

    inline void TestSGDWithMomentum() {
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

    inline void TestAdam() {
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

    inline void TestFiles() {
        std::cout << "TestFiles" << std::endl;
        auto loader = DataLoader("mnist_train.csv", "mnist_test.csv");
        NetworkBuilder b(In{784});
        b.add_layer(Out{20}, ActivationFunction::Sigmoid())
                .add_layer(Out{10}, ActivationFunction::Sigmoid());
        Network network = b.extract();
        FileWriter writer = FileWriter("network.txt");
        writer.write(network);
        FileReader reader = FileReader("network.txt");
        Network new_network = reader.read<Network>();
        std::cout << "Test successfully passed!" << std::endl << std::endl;
    }


    inline void RunALLTests() {
        TestGD();
        TestSGD();
        TestSGDWithMomentum();
        TestAdam();
        TestFiles();
    }
}
