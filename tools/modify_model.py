import torch

def load_and_modify_weights(model_path, save_path):
      # Load the model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Print original keys
    print("Original keys:")
    for key in state_dict.keys():
        print(key)

    # Modify the keys by removing 'backbone' prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_key = key[len('backbone.'):]
        elif key.startswith('decode_head.'):
            new_key = key[len('decode_head.'):]
        elif key.startswith('auxiliary_head.'):
            new_key = key[len('auxiliary_head.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    # Print modified keys
    print("\nModified keys:")
    for key in new_state_dict.keys():
        print(key)

    # Save the modified weights
    # checkpoint['state_dict'] = new_state_dict
    torch.save(new_state_dict, save_path)

def print_model(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    # Print original keys
    print("model")
    for key in state_dict.keys():
        print(key)

if __name__ == "__main__":
    model_path = 'work_dirs/upernet_transnext_tiny_512x512_160k_ade20k_ss/iter_48000.pth'
    save_path = 'pretrained/iter_48k.pth'
    load_and_modify_weights(model_path, save_path)
    print_model(save_path)
    # model_path = 'pretrained/transnext_tiny_224_1k.pth'
    # print_model(model_path)