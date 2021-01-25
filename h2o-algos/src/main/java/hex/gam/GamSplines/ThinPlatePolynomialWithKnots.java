package hex.gam.GamSplines;

import water.MRTask;
import water.MemoryManager;
import water.fvec.Chunk;
import water.fvec.NewChunk;

import static hex.gam.GamSplines.ThinPlatePolynomialBasisUtils.*;
import static hex.genmodel.algos.gam.GamUtilsThinPlateRegression.calculatePolynomialBasis;

public class ThinPlatePolynomialWithKnots extends MRTask<ThinPlatePolynomialWithKnots> {
  final int _weightID;
  final int[][] _polyBasisList;
  final int _M; // size of polynomial basis
  final int _d; // number of predictors used

  public ThinPlatePolynomialWithKnots(int weightID, int[][] polyBasis) {
    _weightID = weightID;
    _d = weightID;
    _polyBasisList = polyBasis;
    _M = polyBasis.length;
  }

  @Override
  public void map(Chunk[] chk, NewChunk[] newGamCols) {
    int numRow = chk[0].len();
    double[] onePolyRow = MemoryManager.malloc8d(_M);
    double[] oneDataRow = MemoryManager.malloc8d(_d);
    for (int rowIndex = 0; rowIndex < numRow; rowIndex++) {
      if (chk[_weightID].atd(rowIndex) != 0) {
        if (checkRowNA(chk, rowIndex)) {
          fillRowOneValue(newGamCols, _M, Double.NaN);
        } else {
          extractOneRowFromChunk(chk, rowIndex, oneDataRow, _d); // extract data to oneDataRow
          calculatePolynomialBasis(onePolyRow, oneDataRow, _d, _M, _polyBasisList);
          fillRowArray(newGamCols, _M, onePolyRow);
        }
      } else {  // set the row to zero
        fillRowOneValue(newGamCols, _M, 0.0);
      }
    }
  }
  
  public static void extractOneRowFromChunk(Chunk[] chk, int rowIndex, double[] oneRow, int d) {
    for (int colInd = 0; colInd < d; colInd++)
      oneRow[colInd] = chk[colInd].atd(rowIndex);
  }
}
