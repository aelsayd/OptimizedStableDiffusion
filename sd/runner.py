import argparse

parser = argparse.ArgumentParser(
                    prog = 'Stable diffusion semi fork',
                    description = 'Mixes most current available huggingface checkpoints into one program that is accesible from the command line and doesnt have a huge amount of dependencies',
                    epilog = 'Lightweight...')
parser.add_argument('-t', "--type", required=True, help="accepts options [gen, inpaint, depth, upscale]")
parser.add_argument('-p', "--prompt", help="Textual prompt, used for diffusion")
parser.add_argument('-i', "--image" , help="Image")
parser.add_argument('-m', "--mask", help="mask")
parser.add_argument('-w', "--width", default=512, help="width of image")
parser.add_argument('-k', "--height", default=512, help="height of image")
parser.add_argument('-r', "--rows", default=1, help="rows")
parser.add_argument('-c', "--columns", default=1, help="columns")
parser.add_argument('-n', "--inference", default=50, help="number of inferences used")
parser.add_argument('-o', "--out", default="images/", help="output path")

args = parser.parse_args()

from main import *
if args.type == "gen":
    generate(args.prompt, int(args.inference), args.out, int(args.width), int(args.height), rows = int(args.rows), columns=int(args.columns))
elif args.type ==  "up":
    upscale(args.prompt, args.image, '{args.out}+gen_{args.prompt}.png')
elif args.type == "in":
    inpaint(args.prompt, args.mask, args.image, '{args.out}+gen_{args.prompt}.png')
elif args.type == "depth":
    depth(args.prompt, args.image, args.out)
