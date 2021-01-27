package hex.genmodel.algos.gam;

import hex.genmodel.ConverterFactoryProvidingModel;
import hex.genmodel.GenModel;
import hex.genmodel.MojoModel;
import hex.genmodel.easy.CategoricalEncoder;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.RowToRawDataConverter;
import hex.genmodel.utils.ArrayUtils;
import hex.genmodel.utils.DistributionFamily;
import hex.genmodel.utils.LinkFunctionType;

import java.util.Map;

import static hex.genmodel.algos.gam.GamUtilsThinPlateRegression.*;
import static hex.genmodel.utils.ArrayUtils.multArray;
import static hex.genmodel.utils.ArrayUtils.nanArray;

public abstract class GamMojoModelBase extends MojoModel implements ConverterFactoryProvidingModel {
  public LinkFunctionType _link_function;
  boolean _useAllFactorLevels;
  int _cats;
  int[] _catNAFills;
  int[] _catOffsets;
  int _nums;
  int _numsCenter;
  double[] _numNAFills;
  double[] _numNAFillsCenter;
  boolean _meanImputation;
  double[] _beta;
  double[] _beta_no_center;
  double[] _beta_center;
  double[][] _beta_multinomial;
  double[][] _beta_multinomial_no_center; // coefficients not centered for multinomial/ordinal
  double[][] _beta_multinomial_center; // coefficients not centered for multinomial/ordinal
  DistributionFamily _family;
  String[][] _gam_columns;
  String[][] _gam_columns_sorted;
  int[] _d;
  int[] _m;
  int[] _M;
  int[] _gamPredSize;
  int _num_gam_columns;
  int[] _bs;
  int[] _bs_sorted;
  int[] _num_knots;
  int[] _num_knots_sorted;
  int[] _num_knots_TP;
  double[][][] _knots;
  double[][][] _binvD;
  double[][][] _zTranspose;
  double[][][] _zTransposeCS;
  String[][] _gamColNames;  // expanded gam column names
  String[][] _gamColNamesCenter;
  String[] _names_no_centering; // column names of features with no centering
  int _totFeatureSize; // Gam Algo predictors numbers that include: predictors, expanded gam columns no centered
  int _betaSizePerClass;
  int _betaCenterSizePerClass;
  double _tweedieLinkPower;
  double[][] _basisVals; // store basis values array for each gam column, avoid new memory allocation per row
  double[][] _hj;   // difference between knot values
  int _numExpandedGamCols; // number of expanded gam columns
  int _numExpandedGamColsCenter; // number of expanded gam columns centered
  int _lastClass;
  int[][][] _allPolyBasisList;
  int _num_TP_col;
  int _num_CS_col;
  // following arrays are pre-allocated to avoid repeated memory allocation per row of scoring
  double[][] _tpRowVals;    // store each row of predictors for each TP smoother
  double[][] _tpDistance;  // store distance measure for each row for all smoothers
  double[][] _tpDistzCS;   // store distance measure * zCS
  double[][] _tpPoly;      // store polynomial basis
  double[][] _tpDistzCSPoly; // concatenate distance measure *zCS + polynomial
  double[][] _tpDistzCSPolyzT; // centered distance measure *zCS + polynomial
  boolean[] _dEven;
  double[] _constantTerms;
  
  GamMojoModelBase(String[] columns, String[][] domains, String responseColumn) {
    super(columns, domains, responseColumn);
  }

  @Override
  public double[] score0(double[] row, double[] preds) {
    if (_meanImputation) {
      imputeMissingWithMeans(row);  // perform imputation for each row
    }
    return gamScore0(row, preds);
  }
  
  void init() {
    if (_num_CS_col > 0) {
      _basisVals = new double[_num_CS_col][]; // for cubic spline smoothers only
      _hj = new double[_num_CS_col][];
      for (int ind = 0; ind < _num_CS_col; ind++) {
        _basisVals[ind] = new double[_num_knots_sorted[ind]];
        _hj[ind] = ArrayUtils.eleDiff(_knots[ind][0]);
      }
    }
    if (_num_TP_col > 0) {
      _tpRowVals = new double[_num_TP_col][];
      _tpDistance = new double[_num_TP_col][];
      _tpDistzCS = new double[_num_TP_col][];
      _tpPoly = new double[_num_TP_col][];
      _tpDistzCSPoly = new double[_num_TP_col][];
      _tpDistzCSPolyzT = new double[_num_TP_col][];
      _dEven = new boolean[_num_TP_col];
      _constantTerms = new double[_num_TP_col];
      for (int index = 0; index < _num_TP_col; index++) {
        int absIndex = index+_num_CS_col;
        _tpRowVals[index] = new double[_d[absIndex]];
        _tpDistance[index] = new double[_num_knots_sorted[absIndex]];
        _tpDistzCS[index] = new double[_num_knots_sorted[absIndex]-_M[index]];
        _tpPoly[index] = new double[_M[index]];
        _tpDistzCSPoly[index] = new double[_num_knots_sorted[absIndex]];
        _tpDistzCSPolyzT[index] = new double[_num_knots_sorted[absIndex]-1];
        _dEven[index] = (_d[absIndex] % 2) == 0;
        _constantTerms[index] = calTPConstantTerm(_m[index], _d[absIndex], _dEven[index]);
      }
    }
    _lastClass = _nclasses - 1;
  }
  
  abstract double[] gamScore0(double[] row, double[] preds);
  
  private void imputeMissingWithMeans(double[] data) {
    for (int ind=0; ind < _cats; ind++)
      if (Double.isNaN(data[ind])) data[ind] = _catNAFills[ind];

    if (data.length == nfeatures()) { // using centered gam cols, nfeatures denotes centered gam columns
      for (int ind = 0; ind < _numsCenter; ind++)
        if (Double.isNaN(data[ind + _cats])) data[ind + _cats] = _numNAFillsCenter[ind];
    } else {
      for (int ind = 0; ind < _nums; ind++) {
        int colInd = ind+_cats;
        if (Double.isNaN(data[colInd])) 
          data[colInd] = _numNAFills[ind];
      }
    }
  }
  
  double evalLink(double val) {
    switch (_link_function) {
      case identity: return GenModel.GLM_identityInv(val);
      case logit: return GenModel.GLM_logitInv(val);
      case log: return GenModel.GLM_logInv(val);
      case inverse: return GenModel.GLM_inverseInv(val);
      case tweedie: return GenModel.GLM_tweedieInv(val, _tweedieLinkPower);
      default: throw new UnsupportedOperationException("Unexpected link function "+_link_function);
    }
  }

  // This method will read in categorical value and adjust for when useAllFactorLevels = true or false
  int readCatVal(double data, int dataIndex) {
    int ival = _useAllFactorLevels ? ((int) data) : ((int) data - 1);
    if (ival < 0)
      return -1;
    ival += _catOffsets[dataIndex];
    return ival;
  }

  // this method will generate the beta*data+intercept
  double generateEta(double[] beta, double[] data) {
    double eta = 0.0;
    int catOffsetLength = _catOffsets.length - 1;
    for (int i = 0; i < catOffsetLength; ++i) {  // take care of contribution from categorical columns
      int ival = readCatVal(data[i], i);
      if ((ival < _catOffsets[i + 1]) && (ival >= 0))
        eta += beta[ival];
    }
    int noff = _catOffsets[_cats] - _cats;
    int numColLen = beta.length - 1 - noff;
    for (int i = _cats; i < numColLen; ++i)
      eta += beta[noff + i] * data[i];
    eta += beta[beta.length - 1]; // add intercept
    return eta;
  }
  
  // check if gamificationis needed.  If all gamified column values are NaN, we need to add gamification.  Otherwise,
  // gamification is already done.
  private boolean gamificationNeeded(double[] rawData, int gamColStart) {
      for (int cind = gamColStart; cind < rawData.length; cind++)
        if (!Double.isNaN(rawData[cind])) {
          return false;
        }
    return true;  
  }
  
  // this method will add to each data row the expanded gam columns with centering
  double[] addExpandGamCols(double[] rawData, final RowData rowData) { // add all expanded gam columns here
    int dataIndEnd = _nfeatures - _numExpandedGamColsCenter; // starting index to fill out the rawData
    if (!gamificationNeeded(rawData, dataIndEnd))
       return rawData;     // already contain gamified columns.  Nothing needs to be done.
    // add expanded gam columns to rowData
    double[] dataWithGamifiedColumns = nanArray(_nfeatures);  // store gamified columns
    System.arraycopy(rawData, 0, dataWithGamifiedColumns, 0, dataIndEnd);
    int tpCounter = 0;
    for (int cind = 0; cind < _num_gam_columns; cind++) { // go through all gam_columns, CS and TP
      if (_bs_sorted[cind] == 0) { // to generate basis function values for cubic regression spline
        Object dataObject = rowData.get(_gam_columns_sorted[cind]); // read predictor column
        double gam_col_data = Double.NaN;
        if (dataObject == null) {  // NaN, skip column gamification
          dataIndEnd += _num_knots_sorted[cind];
          continue;
        } else  // can only test this with Python/R client
          gam_col_data = (dataObject instanceof String) ? Double.parseDouble((String) dataObject) : (double) dataObject;
        GamUtilsCubicRegression.expandOneGamCol(gam_col_data, _binvD[cind], _basisVals[cind], _hj[cind], _knots[cind][0]);
        System.arraycopy(_basisVals[cind], 0, dataWithGamifiedColumns, dataIndEnd, _num_knots_sorted[cind]); // copy expanded gam to rawData
      } else if (_bs_sorted[cind] == 1) { // tp regression
        int relIndex = cind - _num_CS_col;
        String[] gamCols = _gam_columns_sorted[cind];
        double[] gamPred = grabPredictorVals(gamCols, rowData, _tpRowVals[relIndex]); // grabbing multiple predictors
        if (gamPred == null) {
          dataIndEnd += _num_knots_sorted[cind];
          continue;
        }
        calculateDistance(_tpDistance[tpCounter], gamPred, _num_knots_sorted[cind], _knots[cind], _d[tpCounter],
        _dEven[tpCounter], _constantTerms[tpCounter]); // calculate distance between row and knots, result in rowValues
        multArray(_tpDistance[tpCounter], _zTransposeCS[tpCounter], _tpDistzCS[tpCounter]); // distance * zCS
        calculatePolynomialBasis(_tpPoly[tpCounter], gamPred, _d[tpCounter], _M[tpCounter], _allPolyBasisList[tpCounter]);  // generate polynomial basis
        // concatenate distance zCS and poly basis.
        System.arraycopy(_tpDistzCS[tpCounter], 0, _tpDistzCSPoly[tpCounter], 0, _tpDistzCS[tpCounter].length);
        System.arraycopy(_tpPoly[tpCounter], 0, _tpDistzCSPoly[tpCounter], _tpDistzCS[tpCounter].length, _M[tpCounter]);
        multArray(_tpDistzCSPoly[tpCounter], _zTranspose[tpCounter], _tpDistzCSPolyzT[tpCounter]);
        System.arraycopy(_tpDistzCSPolyzT[tpCounter], 0, dataWithGamifiedColumns, dataIndEnd, _num_knots_sorted[cind]);
        tpCounter++;
      } else {
        throw new IllegalArgumentException("spline type not implemented!");
      }
      dataIndEnd += _num_knots_sorted[cind]; 
    }
    return dataWithGamifiedColumns;
  }
  
  double[] grabPredictorVals(String[] gamCols, final RowData rowData, double[] predVals) {
    int numCol = gamCols.length;
    for (int index = 0; index < numCol; index++) {
      Object data = rowData.get(gamCols[index]);
      if (data == null)
        return null;
      predVals[index] = (data instanceof String) ? Double.parseDouble((String) data) : (double) data;
    }
    return predVals;
  }

  @Override
  public RowToRawDataConverter makeConverterFactory(Map<String, Integer> modelColumnNameToIndexMap,
                                                    Map<Integer, CategoricalEncoder> domainMap,
                                                    EasyPredictModelWrapper.ErrorConsumer errorConsumer,
                                                    EasyPredictModelWrapper.Config config) {
    return new GamRowToRawDataConverter(this, modelColumnNameToIndexMap, domainMap, errorConsumer, config);
  }
}
