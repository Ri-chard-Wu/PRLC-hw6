import sys


# print( 'Argument List:', sys.argv)

infile = open(sys.argv[1], "r")
outfile = open(sys.argv[2], "w")


stat_list = [
    "Stall:",
    "gpu_sim_cycle",
    "gpgpu_n_stall_shd_mem"    
]

# keep_list = []

for line in infile:
    for stat in stat_list:
        if stat in line:
            # keep_list.append(line)
            outfile.write(line)



# print("keep_list: ", keep_list)

infile.close()
outfile.close()
