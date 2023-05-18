import torch
import numpy as np
import pickle

class SolarSystemDataset(torch.utils.data.Dataset):
    def __init__(self, partition='train',  train_timesteps=14016, val_timesteps=1752, test_timesteps=1752, timestep_pred=1000):
        self.partition = partition
        solar_system = pickle.load(open('solar/data/solar_system_data.pkl', 'rb'))
        tot_timesteps = solar_system.get_positions().shape[0]
        self.train_timesteps = train_timesteps # 0.8 year of data
        self.val_timesteps = val_timesteps # 0.1 year of data
        self.test_timesteps = test_timesteps # 0.1 year of data
        self.timestep_pred = timestep_pred # Predict 1000 timesteps into the future
        all_positions = solar_system.get_positions()
        all_velocities = solar_system.get_velocities()
        if self.partition == 'train':
            start_t = 0
            end_t = self.train_timesteps
        elif self.partition == 'val':
            start_t = self.train_timesteps
            end_t = self.train_timesteps + self.val_timesteps
        elif self.partition == 'test':
            start_t = self.train_timesteps + self.val_timesteps
            end_t = self.train_timesteps + self.val_timesteps + self.test_timesteps
        else:
            raise Exception("Wrong partition name %s" % self.partition)
        self.pos = all_positions[start_t:end_t]
        self.vel = all_velocities[start_t:end_t]
        self.n_bodies = self.pos.shape[1]
        self.masses = solar_system.get_masses()
        self.edges = self.get_complete_edges(self.n_bodies)
        self.init_pos = self.pos[:-self.timestep_pred]
        self.final_pos = self.pos[self.timestep_pred:]
        self.init_vel = self.vel[:-self.timestep_pred]

    def __getitem__(self, i):
        return self.init_pos[i], self.init_vel[i], self.masses[..., np.newaxis], self.final_pos[i]

    def __len__(self):
        return self.init_pos.shape[0]

    def get_complete_edges(self, n_bodies):
        edges = []
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    edges.append([i, j])
        edges_array = np.array(edges).T
        return edges_array[0], edges_array[1]

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges
