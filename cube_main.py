# from graph import *
# from Algorithm_3 import *
# from Algorithm_2 import *
from Algorithm_1 import *

def main():
    graph = load_graphs('cube_data.txt')
   # sys.stdout=open("out.txt","w") # write ouput into out.txt
    clusters = K_Algorithm(graph)
    #sys.stdout.close()
if __name__ == "__main__":
    main()