import java.io.*;
import java.util.*;

public class hmm {

	//model variables
	/**
	 * A[n][n] : state transition probability distribution, where A[i][j] = Going from state i to state j
	 * B[n][k] : observation symbol probability distribution, where B[i][v] = Probability of observing v at state i
	 * Pi[n]   : initial probability distribution, where Pi[i] = Probability of being at state i at time 1
	 */
	private static double[][] A, B, A_model, B_model;
	private static double[] Pi, Pi_model;
	private static int numStates,numUniqObs;
	
	/**
	 * Main method
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//command : obsv_prob 
		//input   : numberOfStates modelfile(model.txt) testSequence(testdata.txt)
		//return  : probability of a given test sequence given the model, ie P(O|Model)
		//method  : forward procedure
		
		//command : viterbi
		//input   : numberOfStates modelfile(model.txt) testSequence(testdata.txt)
		//function: find best state sequence for a given test sequence
		//return  : best state sequence, as well as P(O|Model) computed over this best state sequence, number of necessary state transitions
		
		//method  : forward procedure

		printInstructions();
		
		String input="";
		String[] inputSplitted;
		
		//prepare to read from cmd line
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		while(true)
		{
			System.out.print("\nPlease enter the command : ");
			try{
				input = reader.readLine();
				if(input.equalsIgnoreCase("e")) //Exit
					break;
				else if(input.equalsIgnoreCase("h")) //Help
					printInstructions();
				else{
					inputSplitted = input.split("\\s+"); //split the input
					if(inputSplitted[0].equalsIgnoreCase("obsv_prob") && inputSplitted.length==4)
						calculateObsvProb(Integer.parseInt(inputSplitted[1]), inputSplitted[2], inputSplitted[3]);
					else if(inputSplitted[0].equalsIgnoreCase("viterbi") && inputSplitted.length==4)
						calculateViterbi(Integer.parseInt(inputSplitted[1]), inputSplitted[2], inputSplitted[3]);
					else if(inputSplitted[0].equalsIgnoreCase("learn") && inputSplitted.length==3)
						learnUsingBaumWelch(Integer.parseInt(inputSplitted[1]), inputSplitted[2]);//numberofstates, datafile
					else
					{
						System.out.println("The command is incorrect.");
						continue;
					}
				}
			}
			catch(Exception e)
			{
				System.out.println("Error while reading from the command line : " + e.toString() + "\n Terminating the program..");
				return;
			}
		}
		System.out.println("\nProgram ended..");
	}
	
	/**
	 * 
	 * @param numOfStates  : number of states of the model
	 * @param dataFileName : name of the data file that contains an observation sequence at each line, 0:correct, 1:incorrect
	 */
	private static void learnUsingBaumWelch(int numOfStates, String dataFileName)
	{
		double up, down, seq_up, seq_down;
		numStates = numOfStates;
		ArrayList<ObservationSequence> obsSequences = readObservationFile(dataFileName);
		numUniqObs = getNumberOfUniqueObs(obsSequences);
		//allocate arrays for model parameters A, B and Pi
		A = new double[numStates][numStates];
		B = new double[numStates][numUniqObs];
		Pi = new double[numStates];
		
		A_model = new double[numStates][numStates];
		B_model = new double[numStates][numUniqObs];
		Pi_model = new double[numStates];
		
		//initializing Pi,randomly // uniform or Pi[0] = 1 and rest is 0
		Random randomGen = new Random();
		double sum=0.;
		for(int i=0; i<numStates; i++)
			sum+=Pi[i]=randomGen.nextInt(98)+1; //between 1 and 99
		for(int i=0; i<numStates; i++)
			Pi[i]/=sum;
		//Pi[0]=.6; Pi[1]=.4;
		//initializing A = uniform distribution
		for(int i=0; i<numStates; i++)
		{
			sum=0.;
			for(int j=0; j<numStates; j++)
				sum+=A[i][j]=randomGen.nextInt(98)+1;
			for(int j=0; j<numStates; j++)
				A[i][j]/=sum;
		}
		//initializing B = uniform distribution		
		for(int i=0; i<numStates; i++)
		{
			sum=0.;
			for(int j=0; j<numUniqObs; j++)
				sum+=B[i][j]=randomGen.nextInt(98)+1;
			for(int j=0; j<numUniqObs; j++)
				B[i][j]/=sum;
		}
		//**********************
		System.out.println("Initial parameters are : ");
		printModel(A,B,Pi);
		
		//now, start training
		//each obs seq will have forward, backward, and c values
		//algorithm stops after a tolerance level between old and new likelihood reached
		double old_likelihood, new_likelihood;
		old_likelihood=new_likelihood=Double.MAX_VALUE;
		double tolerance_level=0.001;
		//STARTING THE ITERATIONS FOR HMM MODEL ESTIMATION
		for(int steps=0; steps<100; steps++)
		{
			ObservationSequence cur=null;
			new_likelihood=0.;
			//calculate vars for each iteration
			for(int i=0; i<obsSequences.size();i++)
			{
				cur = obsSequences.get(i);
				cur.calculateForwards(A, B, Pi);
				cur.calculateBackwards(A, B);			
				cur.calculateUpper(); //gammas and xis
				new_likelihood += cur.calculateProbGivenModel(); // summation of scalings[t]						
			}
			/***compute likelihood***/
			new_likelihood /= (double)obsSequences.size();
			if(Math.abs(old_likelihood-new_likelihood)<=tolerance_level)
			{
				System.out.println("After completing iteration#" + steps + "\nLikelihood = " + new_likelihood);
				break;
			}
			old_likelihood=new_likelihood;
			new_likelihood=0.;
			/***estimate new model parameters***/
			for(int i=0; i<numStates; i++)
				Pi_model[i] = 0.;
			for(int i=0; i<numStates; i++)
			{
				sum=0.;
				for(int k=0; k<obsSequences.size(); k++)
					sum+=(obsSequences.get(k).gammas[0][i]);
				Pi_model[i] = sum/(double)obsSequences.size();
			}
			
			for(int i=0; i<numStates; i++)
			{
				for(int j=0; j<numStates; j++)
				{
					up=down=0.;
					for(int k=0; k<obsSequences.size(); k++)
					{
						seq_up=seq_down=0.;
						cur=obsSequences.get(k);
						for(int t=0; t<cur.length-1; t++)
						{
							seq_up+=(cur.forwards[t][i]*A[i][j]*B[j][cur.observation[t+1]]*cur.backwards[t+1][j]);
							seq_down+=((cur.forwards[t][i]*cur.backwards[t][i])*cur.scalings[t]);
						}
						up+=seq_up;
						down+=seq_down;
					}
					if(up==0 || down==0)
					{
						A_model[i][j]=A[i][j];
					}
					else
						A_model[i][j]=up/down;
				}
			}
			
			//update B
			for(int j=0; j<numStates; j++)
			{
				for(int v=0; v<numUniqObs; v++)
				{
					up=down=0.;
					for(int k=0; k<obsSequences.size(); k++)
					{
						cur=obsSequences.get(k);
						seq_up=seq_down=0.;
						for(int t=0; t<cur.length-1; t++)
						{
							if(cur.observation[t]==v)
								up+=(cur.forwards[t][j]*cur.backwards[t][j]*cur.scalings[t]);
							down+=(cur.forwards[t][j]*cur.backwards[t][j]*cur.scalings[t]);
						}
						up+=seq_up;
						down+=seq_down;
					}
					if(down==0.)
					{
						System.out.println("Div by zero [1]");
						B_model[j][v]=0.;
					}
					else
						B_model[j][v]=up/down;
				}
			}
			A=A_model; B=B_model; Pi=Pi_model;
		}
		printModel(A, B, Pi);
	}
	
	
	private static void printModel(double[][] A, double[][]B, double[] Pi)
	{
		//print all
		System.out.print("Pi =\t[ ");
		for(int i=0; i<numStates; i++)
			System.out.print(Pi[i]+ ", ");
		System.out.print("]\n\n");
		
		System.out.println("A =  ");
		for(int i=0; i<numStates; i++)
		{
			System.out.print("\t");
			for(int j=0; j<numStates; j++)
				System.out.print(A[i][j]+ "  ");
			System.out.println();				
		}
		
		System.out.println("B =  ");
		for(int i=0; i<numStates; i++)
		{
			System.out.print("\t");
			for(int j=0; j<numUniqObs; j++)
				System.out.print(B[i][j]+ "  ");
			System.out.println();				
		}
		System.out.print("\n\n");
	}
	
	
	
	/**
	 * from the given observation list, it extracts the number of unique observations
	 * @param obsList list of observations, given in test file
	 * @return number of unique observations in obsList
	 */
	public static int getNumberOfUniqueObs(ArrayList<ObservationSequence> obsList)
	{
		int[] obs;
		ArrayList<Integer> seenVars = new ArrayList<Integer>();
		for(int i=0;i<obsList.size();i++)
		{
			obs = obsList.get(i).observation;
			for(int j=0; j<obs.length; j++)
			{
				if(!seenVars.contains(obs[j]))
					seenVars.add(obs[j]);
			}
		}
		return seenVars.size();
	}
	
	public static class ObservationSequence{
		int[] observation;
		int length;
		double[][] forwards;
		double[][] backwards;
		double[] scalings;
		double[][] gammas;
		double likelihood;
		double[][][] xi;
		
		public ObservationSequence(String[] obs)
		{
			length = obs.length;
			observation = new int[length];
			for(int i=0; i<length; i++)
				observation[i] = Integer.parseInt(obs[i]);
			forwards = new double[length][numStates];
			backwards = new double[length][numStates];
			scalings = new double[length];
			gammas = new double[length][numStates];
			//xi = new double[length-1][numStates][numStates];
		}
		
		/**calculates the likelihood of this observation sequence given the scaling values***/
		public double calculateProbGivenModel()
		{
			likelihood=0.;
			for(int t=0; t<length; t++)
				likelihood += Math.log((scalings[t]));
			return likelihood;
		}
		
		public void calculateUpper()
		{
			double sum=0.;
			for(int t=0; t<length; t++)
			{
				sum=0.;
				for(int i=0; i<numStates; i++)
				{
					sum+=gammas[t][i]=forwards[t][i]*backwards[t][i];
//					gammas[t][i]*=scalings[t];
				}
				if(sum!=0){
					for(int i=0; i<numStates; i++)
						gammas[t][i]/=sum;
				}
			}
//			double sum = 0.;
//			for(int t=0; t<length; t++)
//			{
//				sum=0.;
//				for(int i=0; i<numStates; i++)
//				{
//					gammas[t][i]=forwards[t][i]*backwards[t][i];
//					sum+=gammas[t][i];
//				}
//				if(sum!=0.)
//				{
//					for(int i=0; i<numStates; i++)
//						gammas[t][i]/=sum;
//				}
//			}
			
			/*
			for(int t=0; t<length-1; t++)
			{
				for(int i=0; i<numStates; i++)
					for(int j=0; j<numStates; j++)
					{
						xi[t][i][j]=forwards[t][i]*A[i][j]*B[j][observation[t+1]]*backwards[t+1][j];
					}
			}
			//estimate gamma
			for(int i=0; i<length; i++)
				for(int j=0; j<numStates; j++)
					gammas[i][j]=0.;
			
			//for t=0..T-1
			for(int t=0; t<length-1; t++)
				for(int i=0; i<numStates; i++)
				{
					for(int j=0; j<numStates; j++)
						gammas[t][i] += xi[t][i][j];
				}
			//for T
			for(int j=0; j<numStates; j++)
				for(int i=0; i<numStates; i++)
					gammas[length-1][j] += xi[xi.length-2][i][j];*/
		}
		
		public void calculateForwards(double[][] A, double[][] B, double[] Pi)
		{
			//for t=0
			scalings[0] = 0.0;
			for(int i=0; i<numStates; i++)
			{
				forwards[0][i]=Pi[i]*B[i][observation[0]];
				scalings[0] += forwards[0][i];
			}
			//scale
			for(int i=0; i<numStates; i++)
				forwards[0][i]/=scalings[0];
			//induction
			for(int t=1; t<length; t++)
			{
				scalings[t]=0.0;
				for(int i=0; i<numStates; i++)
				{
					forwards[t][i]=0.;
					for(int j=0; j<numStates; j++)
						forwards[t][i] += (forwards[t-1][j]*A[j][i]);
					forwards[t][i] *= B[i][observation[t]];
					scalings[t]+=forwards[t][i];
				}
				//scale
				for(int i=0; i<numStates; i++)
					forwards[t][i]/=scalings[t];
			}
			
			//calculateProbGivenModel();
		}
		
		public void calculateBackwards(double[][] A, double[][] B)
		{
			//use scalings
			//at time T with scaling, normally = 1
			for(int i=0; i<numStates; i++)
				backwards[length-1][i]=1./scalings[length-1];
			//induction
			for(int t=length-2; t>=0; t--)
			{
				for(int i=0; i<numStates; i++)
				{
					backwards[t][i]=0.;
					for(int j=0; j<numStates; j++)
						backwards[t][i] += (A[i][j]*B[j][observation[t+1]]*backwards[t+1][j]);
					backwards[t][i]/=scalings[t];
				}
			}
		}
	}
	
	private static ArrayList<ObservationSequence> readObservationFile(String fileName)
	{
		ArrayList<ObservationSequence> observations = new ArrayList<ObservationSequence>();
		try{
			BufferedReader br = new BufferedReader(new FileReader(fileName));
			String line=null;
			String[] splittedLine;
			while((line=br.readLine())!=null)
			{
				splittedLine = line.split("\\s+");
				observations.add(new ObservationSequence(splittedLine));
			}
			br.close();
		}
		catch(Exception e)
		{
			System.out.println("Error while reading " + fileName + " : " + e.toString());
		}
		return observations;
	}
	
	/***ASSIGNMENT 3***/
	
	/**
	 * reads the test sequence data saved in testFile, and returns the test sequence as an integer array
	 * @param testFile : the file name that contains the test sequence
	 * @return an integer array that contains the test sequence
	 */
	public static int[] readTestFile(String testFile)
	{
		String wholeFile,temp;
		String[] splitted; int[] res;
		try{
			BufferedReader reader = new BufferedReader(new FileReader(testFile));
			wholeFile = " ";
			wholeFile=reader.readLine();
			while((temp=reader.readLine())!=null)
			{
				wholeFile.concat(" " + temp);
				temp = reader.readLine();
			}
			reader.close();
		}
		catch(Exception e)
		{
			System.out.println("Error while reading test file " + testFile +" : " + e.toString()); 
			return null;
		}
		splitted = wholeFile.split("\\s+");
		res = new int[splitted.length];
		for(int i=0;i<splitted.length;i++)
			res[i] = Integer.parseInt(splitted[i]); //convert String to int
		return res;
	}
	

	/**
	 * calculates the probability of observation in testFile given the model, in modelFile
	 * using forward procedure. Scaling done. It returns the log probability of observing
	 * the sequence in testFile.
	 *
	 * @param numStates : number of states
	 * @param modelFile : name of the model file 
	 * @param testFile  : name of the file that contains the test sequence
	 */
	public static void calculateObsvProb(int numStates, String modelFile, String testFile)
	{
		//read the model file, in case it is unsuccessful, terminate the function.
		if(!readModelFile(modelFile))
			return;
		//read the observation sequence
		int[] testSeq_int = (readTestFile(testFile));
		if(testSeq_int==null)
			return;
		int T = testSeq_int.length; //T : number of observations in the test sequence, = total time starting at 1
		double[][] alpha = new double[T][numStates]; //the alpha variables
		double[] c=new double[T];//scaling factor c[t] for each time step 0<=t<=T-1
		//calculate alpha at T=1, and the scaling factor
		c[0] = 0.0;
		for(int i=0; i<numStates; i++)
		{
			alpha[0][i] = Pi[i]*B[i][testSeq_int[0]];
			c[0] += alpha[0][i];
		}
		c[0] = 1.0/c[0];
		for(int i=0; i<numStates; i++) //scale
			alpha[0][i]=c[0]*alpha[0][i];
		//calculate alpha till T
		for(int t=1; t<T; t++)
		{
			c[t]=0.0;
			for(int i=0; i<numStates; i++)
			{
				alpha[t][i]=0;
				for(int j=0;j<numStates; j++)
					alpha[t][i] += (alpha[t-1][j]*A[j][i]);
				alpha[t][i] = alpha[t][i]*B[i][testSeq_int[t]];
				c[t]+=alpha[t][i];
			}
			//scale alpha[t]
			c[t]=1.0/c[t];
			for(int s=0;s<numStates;s++)
				alpha[t][s]=c[t]*alpha[t][s];
		}
		//now, calculate the probability from c[t]
		double prob=0.0;
		for(int t=0; t<T; t++)
			prob += Math.log(c[t]);
		prob=-prob;
		System.out.println("log P(O|Model) = " + prob);
	}
	
	/*
	 * finds the best state sequence for a given test sequence
	 * returns the best state sequence, P(Obest|Model), number of necessary state transitions
	 */

	public static void calculateViterbi(int numStates, String modelFile, String testFile)
	{
		//delta [t][i] = best score along a single path, at time t, which accs for the first t observations and ends in state i.
		//phi [t][j] = argument that maximizes delta[t+1][j]

		if(!readModelFile(modelFile))
			return;
		int[] testSeq_int = (readTestFile(testFile));
		if(testSeq_int==null)
			return;
		int T = testSeq_int.length; //T : number of observations in the test sequence, = total time starting at 1
		double[][] delta = new double[T][numStates];
		int[][] phi = new int[T][numStates];
		//initialization, t=0
		for(int i=0; i<numStates; i++)
		{
			delta[0][i] = Math.log(Pi[i])+Math.log(B[i][testSeq_int[0]]);//Pi[i]*B[i][testSeq_int[i]];
			phi[0][i] = 0; 
		}
		//recursion
		for(int t=1; t<T; t++) //for each time step
		{
			for(int j=0; j<numStates; j++) //for each state
			{
				//find the maximum of all delta[t-1][i]*A[i][j]
				double max=delta[t-1][0]+Math.log(A[0][j]); double temp;
				for(int i=0;i<numStates; i++) //calculate delta[t-1][i]+log(a[i,j]) for each 0<=i<N
				{
					temp = delta[t-1][i]+Math.log(A[i][j]);//(delta[t-1][i]*A[i][j]);
					if(temp>max)
					{
						max = temp;
						phi[t][j] = i;
					}
				}
				delta[t][j] = max + Math.log(B[j][testSeq_int[t]]);//max * B[j][testSeq_int[t]];
			}
		}
		//finalize : compute pStar, log P[O|Model], and find the end point of this single path
		double pStar = delta[T-1][0];
		int[] qStar=new int[T]; //all states
		for(int i=0;i<numStates;i++)
		{
			if(delta[T-1][i]>pStar)
			{
				pStar = delta[T-1][i];
				qStar[T-1] = i;
			}
		}
		//backtrack the path and calculate number of state transitions required
		int numStateTransitions = 0;
		for(int t=T-2; t>=0; t--)
		{
			qStar[t] = phi[t+1][qStar[t+1]];
			if(qStar[t+1]!=qStar[t])
				numStateTransitions++;
		}
		
		System.out.println("Best state sequence is : ");
		System.out.print("\t");
		for(int t=0; t<T; t++)
			System.out.print(qStar[t] + " ");
		System.out.println();
		//find P(O,Q|Model) over this sequence
		// = (MULT)i=1:T
		/*double logprob = 0.0;
		double[][] logA = new double[numStates][numStates];
		double[][] logB = new double[numStates][numUniqObs];
		double[] logPi  = new double[numStates];
		for(int i=0; i<numStates; i++)
		{	
			for(int j=0; j<numStates; j++)
				logA[i][j] = Math.log(A[i][j]);
			for(int j=0; j<numUniqObs; j++)
				logB[i][j] = Math.log(B[i][j]);
			logPi[i] = Math.log(Pi[i]);
		}
		logprob = logPi[qStar[0]]+logB[qStar[0]][testSeq_int[0]];
		for(int t=1; t<T; t++) {
			logprob += logA[qStar[t-1]][qStar[t]]+logB[qStar[t]][testSeq_int[t]];
		}*/
		//System.out.println("p* = " + pStar);
		System.out.println("log P(O|Model) over this sequence is = " + pStar);
		
		System.out.println("Number of necessary state transitions = " + numStateTransitions);
	}
	
	/**
	 * initializes the model parameters A, B, and Pi by reading the file named fileName, given as an argument.
	 * @param fileName : name of the file that stores the model parameters A, B, and Pi in specified format.
	 */
	public static boolean readModelFile(String fileName){
		try{
			//open the file
			/*file format:
			A

			a11 a12 ..... a1n
			a21 a22 ..... a2n
			.
			.
			.
			.
			.
			.
			an1 an2 ..... ann
			
			B
			
			b11 b12 .... b1k
			b21 b22 .... b2k
			.
			.
			.
			.
			.
			bn1 bn2 .... bnk
			
			Pi
			
			p1 p2 p3 .... pn
			 */
			BufferedReader reader = new BufferedReader(new FileReader(fileName)); 
			String splitted[];
			numStates=numUniqObs=0;
			//first, we will read A
			if((reader.readLine()).equals("A"))
			{
				//then, this is the state transition matrix
				reader.readLine(); //the empty line
				splitted = (reader.readLine()).split("\\s+"); //read first line and parse
				numStates = splitted.length;
				//create matrix A
				A = new double[numStates][numStates];
				for(int i=0;i<numStates;i++)
					A[i] = new double[numStates];
				for(int i=0; i<numStates; i++)
					A[0][i] = Double.parseDouble(splitted[i]);
				for(int i=1; i<numStates; i++)
				{
					//read the line and parse
					splitted = (reader.readLine()).split("\\s+");
					for(int j=0; j<numStates; j++)
						A[i][j] = Double.parseDouble(splitted[j]);
				}
			}
			reader.readLine(); //read empty line following last line of A matrix
			if((reader.readLine()).equals("B"))
			{
				reader.readLine(); //empty line
				splitted = (reader.readLine()).split("\\s+");
				numUniqObs = splitted.length;
				//create matrix B
				B = new double[numStates][numUniqObs];
				for(int i=0;i<numUniqObs;i++)
					B[0][i] = Double.parseDouble(splitted[i]);
				for(int i=1; i<numStates; i++)
				{
					//read the line and parse
					splitted = (reader.readLine()).split("\\s+");
					for(int j=0; j<numUniqObs; j++)
						B[i][j] = Double.parseDouble(splitted[j]);
				}
			}
			reader.readLine();
			if((reader.readLine()).equals("Pi"))
			{
				reader.readLine(); //empty line
				Pi = new double[numStates];
				splitted = (reader.readLine()).split("\\s+");
				for(int i=0;i<numStates;i++)
					Pi[i] = Double.parseDouble(splitted[i]);
			}
			reader.close();
		}
		catch(Exception e)
		{	
			System.out.println("Error while reading model in file " + fileName.toString());
			return false;
		}
		return true;
	}
	
	/**
	 * prints the instructions explaining how to use the program.
	 */
	public static void printInstructions()
	{
		System.out.println("How to use? ");
//		System.out.println("\tEnter \n\t     obsv_prob numberOfStates modelFile testSequenceFile \n\tto compute probability of the test sequence given as the 3rd argument, computed by forward procedure.");
//		System.out.println("\tEnter \n\t     viterbi numberOfStates modelFile testSequenceFile \n\tto find the best state sequence(Q), and compute the P(O|Model) over this sequence Q along with number of necessary state transitions.");
		System.out.println("\tEnter \n\t	 learn numberOfStates dataFile \n\tto train HMM using Baum-Welch method, and output model to HMMdescription.txt");
		System.out.println("\tEnter h to display instructions.");
		System.out.println("\tEnter e to exit.");
	}
}
