# -*- Mode: Perl -*-
# t/02_txtprob.t: test text-probability

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

# load common subs
use Test;
do "$TEST_DIR/common.plt";
use PDL;
use PDL::HMM;

BEGIN { plan tests=>8, todo=>[]; }

## test model 1:
sub testmodel1 {
  ##-- model
  $a = pdl(double, [[0,   1/2, 1/2],
		    [1/3, 1/3, 1/3],
		    [1/3, 1/3, 1/3]])->log;

  $b = pdl(double, [[1,   0,   0],
		    [0,   1/2, 1/2],
		    [0,   1/2, 1/2]])->log;

  $pi = pdl(double, [1,   0,   0])->log;

  $o = pdl(long, [0,1,2,0]);

  ##-- forward (alpha)
  $fw = hmmfw($a,$b,$pi,$o);
  $fwtp = $fw->slice(",".($fw->dim(1)-1))->logsumover;

  $fw_want = pdl(double, [[   1,    0,    0],
			  [   0,  1/4,  1/4],
			  [   0, 1/12, 1/12],
			  [1/18,    0,    0]])->log;
  $fwtp_want = log(1/18);


  ##-- backward (beta)
  $bw   = hmmbw($a,$b,$o);
  $bwtp = logsumover($pi+$bw->slice(",0"));

  $bw_want  = pdl(double, [[1/18,  1/27,  1/27],
			   [ 1/6,   1/9,   1/9],
			   [   0,   1/3,   1/3],
			   [   1,     1,     1]])->log;
}


## test model 2:
sub testmodel2 {
  ##-- model
  $a = pdl(double, [[0,   1/2, 1/2],
		    [1/2,   0, 1/2],
		    [1/2, 1/2,   0]])->log;
  $b = pdl(double, [[1,   0,   0],
		    [0,   1,   0],
		    [0,   0,   1]])->log;

  $pi = pdl(double, [1,0,0])->log;

  ##-- input
  $o = pdl(long, [0,1,2,0]);


  ##-- forward
  $fw = hmmfw($a,$b,$pi,$o);
  $fwtp = logsumover($fw->slice(",".($fw->dim(1)-1)));

  $fw_want = pdl(double, [[  1,   0,   0],
			  [  0, 1/2,   0],
			  [  0,   0, 1/4],
			  [1/8,   0,   0]])->log;

  $fwtp_want = log(1/8);

  ##-- backward
  $bw   = hmmbw($a,$b,$o);
  $bwtp = logsumover($pi+$bw->slice(",0"));

  $bw_want = pdl(double, [[1/8,   0, 1/8],
			  [1/4, 1/4,   0],
			  [  0, 1/2, 1/2],
			  [  1,   1,   1]])->log;

}



##-- 1--4: model 1
testmodel1;
isok("model-1: alpha", all($fw->approx($fw_want,1e-6)));
isok("model-1: beta",  all($bw->approx($bw_want,1e-6)));
isok("model-1: txtprob", all($fwtp->approx($fwtp_want,1e-6)));
isok("model-1: alpha~beta", all($bwtp->approx($fwtp,1e-6)));

##-- 5--8: model-2
testmodel2;
isok("model-2: alpha", all($fw->approx($fw_want,1e-6)));
isok("model-2: beta",  all($bw->approx($bw_want,1e-6)));
isok("model-2: txtprob", all($fwtp->approx($fwtp_want,1e-6)));
isok("model-2: alpha~beta", all($bwtp->approx($fwtp,1e-6)));


print "\n";
# end of t/XX_yyyy.t

