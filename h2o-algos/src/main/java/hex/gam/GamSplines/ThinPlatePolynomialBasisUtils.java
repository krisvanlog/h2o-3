package hex.gam.GamSplines;

import Jama.Matrix;
import Jama.QRDecomposition;
import water.fvec.Chunk;
import water.fvec.NewChunk;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static hex.gam.GAMModel.GAMParameters;
import static hex.util.LinearAlgebraUtils.generateOrthogonalComplement;
import static water.util.ArrayUtils.maxValue;

// This class contains functions that perform different functions for generating the thin plate regression splines
// and the polynomial basis functions
public class ThinPlatePolynomialBasisUtils {
  /***
   * For thin plate regression, given d (number of predictors for a smooth), it will return m where (m-1) is the
   * maximum polynomial degree in the polynomial basis functions.  The formula used is m = ceiling of (d+1)/2+1
   *
   * @param d : integer denoting number of predictors for thin plate regression smooth.
   * @return m : integer denoting the maximum polynomial degree + 1 for polynomial basis function.
   */
  public static int calculatem(int d) {
    return ((int) Math.floor((d+1.0)*0.5))+1;
  }

  public static int calculateM(int d, int m) {
    int topComb = d+m-1;
    return hex.genmodel.utils.MathUtils.combinatorial(topComb, d);
  }

  /***
   * This method, given number of predictors in the smooth d, number of polynomials in the polynomial basis m, will
   * generate a list of integer array specifying for each predictors the degree that predictor will have.  For instance,
   * if for a predictor, the degree is 0, a constant 1 is used.  If for a particular predictor, the degree is 2, 
   * predictor*predictor is used.
   * 
   * @param d
   * @param m
   * @return
   */
  public static List<Integer[]> findPolybasis(int d, int m) {
    int polyOrder = m-1;
    int[] possibleDegree = new int[polyOrder];
    for (int index = 1; index < m; index++) // generate all polynomial orders
      possibleDegree[index-1] = index;
    Integer[] basisPolyOrder = new Integer[d];  // store one combination
    List<Integer[]> totPolyBasis = new ArrayList<>();  // store all combination 
    for (int degree : possibleDegree) {
      ArrayList<int[]> oneCombo = new ArrayList<>();
      findOnePerm(degree, possibleDegree, 0, oneCombo, null);
      mergeCombos(oneCombo, basisPolyOrder, possibleDegree, totPolyBasis);// merge all combos found for all possibleDegree
    }
    return findAllPolybasis(totPolyBasis);
  }

  /***
   * For each list in onePolyBasis, we still need to find all the permutations for that list.  In addition, we need to
   * add the combination for the 0th order as well.  For instance, if the list contains {0,0,1}, we need to add to that
   * the lists {0,1,0} and {1,0,0} as well.
   * @param onePolyBasis
   * @return
   */
  public static List<Integer[]> findAllPolybasis(List<Integer[]> onePolyBasis) {
    int listSize = onePolyBasis.size();
    List<Integer[]> allPermutes = new ArrayList<>();
    for (int index = 0; index < listSize; index++) {
      Integer[] oneBasis = onePolyBasis.get(index);
      int[] freqTable = generateOrderFreq(oneBasis);  // find polynomial basis order and count
      ArrayList<ArrayList<Integer>> basisPermuations = new ArrayList<>();
      ArrayList<Integer> prefix = new ArrayList<>();
      findPermute(freqTable, prefix, oneBasis.length, basisPermuations);
      addPermutationList(allPermutes, basisPermuations);
    }
    // add the list with all zeros
    Integer[] allZeros = new Integer[onePolyBasis.get(0).length];
    for (int index = 0; index < allZeros.length; index++)
      allZeros[index] = 0;
    allPermutes.add(0, allZeros); // add all zero to the front
    return allPermutes;
  }
  
  public static void addPermutationList(List<Integer[]> onePolyBasis, ArrayList<ArrayList<Integer>> permute1Basis) {
    for (ArrayList<Integer> onePermute : permute1Basis) {
      Integer[] oneCombo = onePermute.toArray(new Integer[0]);
      onePolyBasis.add(oneCombo);
    }
  }
  
  public static void findPermute(int[] freqMap, ArrayList<Integer> prefix, int remaining,
                                 ArrayList<ArrayList<Integer>> basisPerm) {
    if (remaining == 0) { // done with choosing all permutation
      basisPerm.add(prefix);
    } else {
      for (int index=0; index < freqMap.length; index++) {
        int val = freqMap[index];
        if (val > 0) {
          freqMap[index]--;
          ArrayList<Integer> newPrefix = new ArrayList<>(prefix);
          newPrefix.add(index);
          findPermute(freqMap, newPrefix, remaining-1, basisPerm);
          freqMap[index] = val;
        }
      }
    }
  }
  
  public static int[] generateOrderFreq(Integer[] oneBasis) {
    int maxVal = maxValue(oneBasis);
    int[] mapFreq = new int[maxVal+1];
    for (int val : oneBasis)
      mapFreq[val]++;
    return mapFreq;
  }
  
  public static void mergeCombos(ArrayList<int[]> oneCombo, Integer[] basisOrder, int[] polyBasis, List<Integer[]> polyBasisSet) {
    for (int[] oneList : oneCombo) {
      Arrays.fill(basisOrder, 0);
      expandCombo(oneList, polyBasis, basisOrder);
      polyBasisSet.add(basisOrder.clone());
    }
  }

  /***
   * Given a combo found by findOnePerm say for d = 5, m = 4, for degree = 1 to m-1 (3 in this case).  The basis poly
   * is {3,2,1}.  The returned list of combo are: {{0, 0, 1}, {0, 1, 0}, {0, 0, 2}, {1,0,0}, {0,1,1}, {0,0,3}}.
   * However, we need to convert this list back to the perspective of the predictors.  In this case, we have 5 
   * predictors.  This function will translate the list from findOnePerm to the perspective of the predictors.  For 
   * instance, for degree = 1, we need to have one predictive to have degree of 1 and hence the list should be 
   * {1, 0, 0, 0, 0}.  For degree = 2, we can have one predictor taking degree of 2 or two predictors each taking 
   * degree 1.  The same applies to the other degrees.  Hence, this function will return the following list: 
   * {{1, 0, 0, 0, 0}, {0, 2, 0, 0, 0}, {1, 1, 0, 0, 0}, {0, 0, 3, 0, 0}, {1, 2, 0, 0, 0} and {1, 1, 1, 0, 0}}.
   * @param oneList
   * @param polyBasis
   * @param basisOrder
   */
  public static void expandCombo(int[] oneList, int[] polyBasis, Integer[] basisOrder) {
    int expandIndex = 0;
    for (int index = 0; index < polyBasis.length; index++) {
      int count = 0;
      if (oneList[index] == 0) {
        basisOrder[expandIndex++] = 0;
      } else {
        while (count < oneList[index]) {
          basisOrder[expandIndex++] = polyBasis[index];
          count++;
        }
      }
    }
  }

  /***
   * For a fixed degree specified as totDegree, specified a set of combination of polynomials to achieve the totDegree.
   * For instance, if degreeCombo = {1,2,3} and the totDegree is 1. There is only one way to achieve it by the array
   * {1,0,0}.  If totDegree = 2, there are two arrays that will work: {0,1,0} or {2,0,0}.  If totDegree = 3, there will
   * be 3 arrays that will work {3,0,0}, {1,1,0}, {0,0,1}.
   * 
   * @param totDegree : integer representing degree of polynomial basis
   * @param degreeCombo : degrees allowed for polynomial basis
   * @param index
   * @param allCombos
   * @param currCombo
   */
  public static void findOnePerm(int totDegree, int[] degreeCombo, int index, ArrayList<int[]> allCombos,
                                 int[] currCombo) {
    if (totDegree == 0) {
      if (currCombo != null)
        allCombos.add(currCombo.clone());
      return;
    } else if (totDegree < 0 || index >= degreeCombo.length) { // last added value too big
      return;
    }  else {
      int totPass = totDegree / degreeCombo[index];
      int degreeCount = 0;
      if (currCombo == null)
        currCombo = degreeCombo.clone();
      while (degreeCount <= totPass) {
        setCombo(currCombo, index, degreeCount);
        findOnePerm(totDegree - degreeCount * degreeCombo[index], degreeCombo, index + 1,
                allCombos, currCombo);
        degreeCount++;
      }
    }
  }

  public static void setCombo(int[] currCombo, int index, int degreeCount) {
    currCombo[index] = degreeCount;
    int combSize = currCombo.length;
    for (int tempIndex = index+1; tempIndex < combSize; tempIndex++) 
      currCombo[tempIndex] = 0;
  }
  
  public static double[][] generateStarT(double[][] knots, List<Integer[]> polyBasisDegree) {
    int numKnots = knots[0].length;
    int M = polyBasisDegree.size();
    int d = knots.length;
    double[][] starT = new double[numKnots][M];
    for (int rowInd = 0; rowInd < numKnots; rowInd++) {
      for (int polyBasisInd = 0; polyBasisInd < M; polyBasisInd++) {
        Integer[] oneBasis = polyBasisDegree.get(polyBasisInd);
        double polyBasisVal = 1.0;
        for (int predInd = 0; predInd < d; predInd++) {
          polyBasisVal *= Math.pow(knots[predInd][rowInd], oneBasis[predInd]);
        }
        starT[rowInd][polyBasisInd] = polyBasisVal;
      }
    }
    return starT;
  }

  public static double[][]extractLastKMinuMCols(double[][] qMat, int rowStart, int rowEnd) {
    int nCol = rowEnd - rowStart;
    int nRow = rowEnd;
    double[][] zCS = new double[nRow][nCol];
    for (int index = 0; index < rowEnd; index++) {
      System.arraycopy(qMat[index], rowStart, zCS[index], 0, nCol);
    }
    return zCS;
  }
  public static void fillRowOneValue(NewChunk[] newChk, int colWidth, double fillValue) {
    for (int colInd = 0; colInd < colWidth; colInd++)
      newChk[colInd].addNum(fillValue);
  }

  public static void fillRowArray(NewChunk[] newChk, int colWidth, double[] fillValue) {
    for (int colInd = 0; colInd < colWidth; colInd++)
      newChk[colInd].addNum(fillValue[colInd]);
  }

  public static boolean checkRowNA(Chunk[] chk, int rowIndex) {
    int numCol = chk.length;
    for (int colIndex = 0; colIndex < numCol; colIndex++) {
      if (Double.isNaN(chk[colIndex].atd(rowIndex)))
        return true;
    }
    return false;
  }
  
  public static String genThinPlateNameStart(GAMParameters parms, int gamColIndex) {
    StringBuffer colNameStub = new StringBuffer();
    for (int gColInd = 0; gColInd < parms._gam_columns_sorted[gamColIndex].length; gColInd++) {
      colNameStub.append(parms._gam_columns_sorted[gamColIndex][gColInd]);
      colNameStub.append("_");
    }
    colNameStub.append(parms._bs_sorted[gamColIndex]);
    colNameStub.append("_");
    return colNameStub.toString();
  }
  
  public static String[] extractColNames(String[] src, int srcStart, int destStart, int length) {
    String[] distanceColNames = new String[length];  // exclude the polynomial basis names
    System.arraycopy(src, srcStart, distanceColNames, destStart, length);
    return distanceColNames;
  }
  
  public static double[][] genZCS(GAMParameters parms, int numVec, int vecSize, double[][] starT) {
    Matrix starTMat = new Matrix(starT);        // generate Zcs as in 3.3
    QRDecomposition starTMat_qr = new QRDecomposition(starTMat);
    double[][] qMat = starTMat_qr.getQ().getArray();
    double[][] zCS = generateOrthogonalComplement(qMat, numVec, parms._seed);
    return zCS;
  }
  
  public static int[][] convertList2Array(List<Integer[]> list2Convert, int M, int d) {
    int[][] polyBasisArr = new int[M][d];
    for (int index = 0; index < M; index++) {
      List<Integer> oneList = Arrays.asList(list2Convert.get(index));
      polyBasisArr[index] = oneList.stream().mapToInt(Integer::intValue).toArray();
    }
    return polyBasisArr;
  }

  public static double[][] allocate2DArr(int firstDim, int[] secondDim) {
    double[][] arrs = new double[firstDim][];
    for (int index=0; index < firstDim; index++)
      arrs[index] = new double[secondDim[index]];
    return arrs;
  }
  
}
