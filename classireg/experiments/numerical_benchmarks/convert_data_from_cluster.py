import sys
from classireg.utils.parse_data_collection import convert_from_cluster_data_to_single_file

def main(which_obj,which_acqui,Nrepetitions) -> None:
	convert_from_cluster_data_to_single_file(which_obj=which_obj,which_acqui=which_acqui,Nrepetitions=Nrepetitions)

if __name__ == "__main__":

	if len(sys.argv) != 4:
		raise ValueError("Required input arguments: <ObjFun> <Algorithm> <Nrepetitions> ")

	ObjFun 	= sys.argv[1]
	which_acqui = sys.argv[2]
	Nrepetitions = int(sys.argv[3])

	main(which_obj=ObjFun,which_acqui=which_acqui,Nrepetitions=Nrepetitions)