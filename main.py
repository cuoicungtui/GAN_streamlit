from lib import  Generator, Args,CLIPText
import torch, transformers
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import  utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st

text_encoder = CLIPText(None)

args = Args()

args.add('device','cuda')
args.add('size', 128)
args.add('r1',10)
args.add('d_reg_every', 16)
args.add('ckpt','\245000.pt')
args.add('channel_multiplier', 2)
args.add('augment_p', 0)
args.add('ada_target', 0.6)
args.add('ada_length', (500 * 1000))
args.add('ada_every', 256)

g_ema = Generator(
    args.size, args.latent, args.n_mlp, args.tin_dim, args.tout_dim,
    channel_multiplier=args.channel_multiplier, use_multi_scale=args.use_multi_scale,
    use_text_cond=args.use_text_cond, use_self_attn=args.g_use_self_attn,
)

ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    g_ema.load_state_dict(ckpt["g_ema"])
    g_ema.eval()

def gen_image(input_text):
    sample_t = text_encoder(input_text)
    sample_z = torch.randn(1, args.latent)
    sample, _ = g_ema([sample_z], sample_t)
    utils.save_image(
        sample[-1], f"sample.png",
        nrow=int(1), normalize=True, value_range=(-1, 1),
        )



# input_text = "A black and white drawing of a dog running"
# gen_image(input_text)



with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if submitted :
       gen_image(text)
       st.image('sample.png')
