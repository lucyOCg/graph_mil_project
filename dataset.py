import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np 
import os
from tqdm import tqdm
import h5py
        
def coords_adjacent2(coords, dictionary):
    "Function to check if patches are adjacent, using patch coordinates as input"
    edge_indices_1 = []
    x1, y1 = coords
    x2 = x1 + 256
    patch = dictionary.get((x2, y1))
    if patch != None:
        edge_indices_1 += [(dictionary[(x1,y1)],patch)]
        edge_indices_1 += [(patch,dictionary[(x1,y1)])] #added for opposite direction graph connection
    x2 = x1 - 256
    patch = dictionary.get((x2, y1))
    if patch != None:
        edge_indices_1 += [(dictionary[(x1,y1)],patch)]
        edge_indices_1 += [(patch,dictionary[(x1,y1)])] #added for opposite direction graph connection 
    y2 = y1 + 256
    patch = dictionary.get((x1, y2))
    if patch != None:
        edge_indices_1 += [(dictionary[(x1,y1)],patch)]
        edge_indices_1 += [(patch,dictionary[(x1,y1)])] #added for opposite direction graph connection
    y2 = y1 - 256
    patch = dictionary.get((x1, y2))
    if patch != None:
        edge_indices_1 += [(dictionary[(x1,y1)],patch)]
        edge_indices_1 += [(patch,dictionary[(x1,y1)])] #added for opposite direction graph connection
    return edge_indices_1        

        
        
def get_patches_from_path(path):
    #print('path:', path)
    patch_file = h5py.File(path, "r")
    #print(patch_file['features'])
    return patch_file


print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

"""
!!!
NOTE: This file was replaced by dataset_featurizer.py
but is kept to illustrate how to build a custom dataset in PyG.
!!!
"""


class PatchDataset(Dataset):
    def __init__(self, root, filename, test=False, val=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.val = val
        self.filename = filename
        super(PatchDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.val:
            return [f'data_val_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, patches in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            img_obj = get_patches_from_path(patches["patch_paths"]) #make sure function doesn't need to be a class one
            # Get node features
            node_feats = self._get_node_features(img_obj)
            # # Get edge features
            # edge_feats = self._get_edge_features(mol_obj)
            # Get adjacency info
            edge_index = self._get_adjacency_info(img_obj)
            # Get labels info
            slide_id = self._get_slide_id(patches["slide_id"])#new to get saved in right format
            label = self._get_labels(patches["label"])

            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        #edge_attr=edge_feats,
                        y=label,
                        slide_id = slide_id
                        #smiles=mol["smiles"]
                        ) 
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            elif self.val:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_val_{index}.pt'))            
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))
                #torch.save(data, 
                    #os.path.join(self.processed_dir, 
                                 #f'{slide_id}.pt'))
                                 
    def _get_node_features(self, h5_file):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []
        img_features = h5_file['features']
        for patch in img_features:
            all_node_feats.append(patch)

        #for atom in mol.GetAtoms():
            #node_feats = []
            ## Feature 1: Atomic number        
            #node_feats.append(atom.GetAtomicNum())
            ## Feature 2: Atom degree
            #node_feats.append(atom.GetDegree())
            ## Feature 3: Formal charge
            #node_feats.append(atom.GetFormalCharge())
            ## Feature 4: Hybridization
            # #node_feats.append(atom.GetHybridization())
            # # Feature 5: Aromaticity
            # node_feats.append(atom.GetIsAromatic())
            # # Feature 6: Total Num Hs
            # node_feats.append(atom.GetTotalNumHs())
            # # Feature 7: Radical Electrons
            # node_feats.append(atom.GetNumRadicalElectrons())
            # # Feature 8: In Ring
            # node_feats.append(atom.IsInRing())
            # # Feature 9: Chirality
            # node_feats.append(atom.GetChiralTag())

            # Append node features to matrix
            #all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    # def _get_edge_features(self, mol):
    #     """ 
    #     This will return a matrix / 2d array of the shape
    #     [Number of edges, Edge Feature size]
    #     """
    #     all_edge_feats = []

    #     for bond in mol.GetBonds():
    #         edge_feats = []
    #         # Feature 1: Bond type (as double)
    #         edge_feats.append(bond.GetBondTypeAsDouble())
    #         # Feature 2: Rings
    #         edge_feats.append(bond.IsInRing())
    #         # Append node features to matrix (twice, per direction)
    #         all_edge_feats += [edge_feats, edge_feats]

    #     all_edge_feats = np.asarray(all_edge_feats)
    #     return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, h5_file):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        img_coords = h5_file['coords']
        img_coords_list = []
        for coord in img_coords:
            img_coords_list.append(tuple(coord))
        
        idx_list = list(range(len(img_coords_list)))
        coord_dict = dict(zip(img_coords_list,idx_list))
        #print(idx_list)
        edge_indices = []
        for coords in img_coords_list:
        #print(patch)
            edge_indices += coords_adjacent2(coords, coord_dict)
        
        
        #for patch in idx_list:
            #for patch_2 in idx_list:
                #x1, y1 = img_coords_list[patch]
                #x2, y2 = img_coords_list[patch_2]
                #coords = coords_adjacent(x1, y1, x2, y2, patch, patch_2)
                #if coords != 'nothing':
                    #edge_indices.append(tuple(coords))
            #print(graph_edges)


        #edge_indices = []
        #for bond in mol.GetBonds():
            #i = bond.GetBeginAtomIdx()
            #j = bond.GetEndAtomIdx()
            #edge_indices += [[i, j], [j, i]]
        #print(edge_indices)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        #print('EDGE INDICES:', edge_indices)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def _get_slide_id(self, slide_id):
        label = np.asarray([slide_id])
        return str(slide_id)       
    

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        self.data = pd.read_csv(self.raw_paths[0])
        slide_id = self.data["slide_id"][idx]
        data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        elif self.val:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_val_{idx}.pt'))        
        
        #else:
            #data = torch.load(os.path.join(self.processed_dir, 
                                 #f'{slide_id}.pt'))                                   
        return data
        
#val_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/", filename="graph_raw_10x_res18_val.csv")        
#train_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/", filename = 'graph_raw_10x_res18_train.csv' )
#test_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/", filename="graph_raw_10x_res18_test.csv", test=True)


#print('dataset length', len(dataset))

#print('num_edge_features', dataset.num_features)


#print('dataset.num_node_features', dataset.indices)

#print(dataset.num_edge_features)

#print(dir(dataset))
