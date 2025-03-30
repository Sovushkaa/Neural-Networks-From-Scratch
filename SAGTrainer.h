#include "DataLoader.h"
#include "Network.h"

namespace CNetworks {
    enum Epoch_number : Index;
    enum Batch_size : Index;
    class SAGTrainer {
    public:
        SAGTrainer(DataLoader &&loader, LossFunction &&loss,
                   Epoch_number epoch_number, Batch_size batch_size,
                   double learning_rate);

        Network Train(Network &&net);

    private:
        void TrainOneEpoch(std::vector<Layer> &layers);

        struct Metrics {
            double average_mistake;
            double success_percent;
        };

        Metrics ComputeMetrics(const Network &net) const;
        void PrintMetrics(const Metrics &metrics) const;

        DataLoader loader_;
        LossFunction loss_;
        Index epoch_number_;
        Index batch_size_;
        double learning_rate_;
    }; // namespace CNetworks
} // namespace CNetworks
