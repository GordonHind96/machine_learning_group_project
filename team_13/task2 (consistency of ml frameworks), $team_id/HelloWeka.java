package weka.api;
import java.awt.List;

//written by Brian de Buiteach

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import weka.*;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class HelloWeka {

	ArrayList<ArrayList<String>> fullList;
	public void HelloWeka(){
		fullList = new ArrayList<ArrayList<String> >();
	}



	private void generateCsvFile(String fileName) {
		FileWriter writer = null;
		Collections.shuffle(fullList);
		try{
			writer = new FileWriter(fileName);

			for(ArrayList<String> currentRowList : fullList){
				for(String str : currentRowList){
					if(str.equals("1")){
						writer.append("0");
					}
					else if(str.equals("2")){
						writer.append("1");
					}
					else{
						writer.append(str);
					}
					writer.append(",");
				}
				writer.append("\n");
			}

			System.out.println("CSV FILE CREATED - MAKE SURE YOU CAN FIND IT MATE");

		}catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				writer.flush();
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}


	private void splitCSV(String name1,String name2,double train,double test) {
		FileWriter writer1 = null;
		FileWriter writer2 = null;
		int index = 0;
		int len = fullList.size();
		try{
			writer1 = new FileWriter(name1);


			while(index < (len * train) ){
				ArrayList<String> curRow = fullList.get(index);
				index ++;
				for(String str : curRow){
					writer1.append(str);
					writer1.append(",");
				}
				writer1.append("\n");
			}

			System.out.println("1st one made - CSV FILE CREATED - MAKE SURE YOU CAN FIND IT MATE");

		}catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				writer1.flush();
				writer1.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		// test data
		try{
			writer2 = new FileWriter(name2);



			while(index < len  ){
				ArrayList<String> curRow = fullList.get(index);
				index ++;
				for(String str : curRow){
					writer2.append(str);
					writer2.append(",");
				}
				writer2.append("\n");
			}

			System.out.println("2nd CSV FILE CREATed");

		}catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				writer2.flush();
				writer2.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}




	}




	public void readInData(String path){
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			StringBuilder sb = new StringBuilder();
			String line = br.readLine();

			ArrayList<String> currentRow = new ArrayList<String>();


			int i=0;
			while (line != null) {
				i = 0;
				char [] tempCharAr = line.toCharArray();
				int length = tempCharAr.length;
				String tempStr;
				tempStr="";
				while(i<length){

					String str = Character.toString(tempCharAr[i]);
					tempStr = tempStr.concat(str);

					if(i+1 <length && tempCharAr[i+1]=='\t'){
						currentRow.add(tempStr);
						i+=2; // to skip over the \t and stay in the loop
						tempStr=""; //reset ever time new string added
					}
					else if(i == length-1 ){
						currentRow.add(tempStr);
						break;
					}
					else
						i++;

				}

				if(fullList == null){
					fullList =  new ArrayList<ArrayList<String> >();
				}

				fullList.add(currentRow);
				currentRow = new ArrayList<String>();



				sb.append(line);
				sb.append(System.lineSeparator());

				line = br.readLine();
			}//finished reading the text file

			br.close();
			System.out.println(sb.toString());


		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("file most liekluy not found");
			e.printStackTrace();
		}
	}



	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("hello weka");
		String path = "C:\\Users\\bboyd\\Documents\\college - 4th year\\Machine Learning\\machine_learning_group_project\\team_13\\datasets\\Skin_NonSkin.txt";

		HelloWeka h = new HelloWeka();
		h.readInData(path);
		h.generateCsvFile("skin.csv");

		//h.splitCSV("skin_train.csv", "skin_test.csv", .7, .3);


		//CSVLoader loader = new CSVLoader();
		
	}

}
