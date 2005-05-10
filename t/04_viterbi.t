# -*- Mode: Perl -*-
# t/04_viterbi.t: test Viterbi algorithm

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

# load common subs
use Test;
do "$TEST_DIR/common.plt";
use PDL;
use PDL::HMM;

BEGIN { plan tests=>3, todo=>[]; }

##-- test model 1:
sub testmodel1 {
  ##-- a
  $a = pdl(double, [[0,   1/4, 1/4],
		    [1/2, 1/2, 1/4],
		    [1/4, 1/4, 1/2]])->log;

  ##-- b
  $b = pdl(double, [[1,   0,   0],
		    [0, 3/4, 1/4],
		    [0, 1/4, 3/4]])->log;

  ##-- pi
  $pi = pdl(double,[1,0,0])->log;
}

##-- tests: model 1

sub vtest {
  ($delta,$psi) = hmmviterbi($a,$b,$pi, $o);
  $path = hmmpath($psi, sequence(long,$delta->dim(0)));
}

##-- 1--3: expect for o=[0,1,2,0]
testmodel1();
$o = pdl(long,[0,1,2,0]);
vtest();

isok("delta1", all($delta->approx( pdl(double, [[1,        0,    0],
						[0,      1/2, 1/12],
						[0,     1/16, 3/32],
						[3/128,    0,    0]])->log, 1e-6 )));

isok("psi1", all($psi == pdl(long, [[0,0,0],
				    [0,0,0],
				    [1,1,1],
				    [2,1,2]])));

isok("path1", all($path == pdl(long, [[0,1,2,0],
				      [0,1,1,1],
				      [0,1,2,2]])));

print "\n";
# end of t/XX_yyyy.t

