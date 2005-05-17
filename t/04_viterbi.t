# -*- Mode: Perl -*-
# t/04_viterbi.t: test Viterbi algorithm

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

# load common subs
use Test;
do "$TEST_DIR/common.plt";
use PDL;
use PDL::HMM;

BEGIN { plan tests=>6, todo=>[]; }

##-- test model 1:
sub testmodel1 {
  $pi = pdl(double,  [.6, .4])->log;

  $a = pdl(double,   [[.5, .2],
		      [.3, .5]])->log;

  $omega = pdl(double,[.2, .3])->log;

  $b = pdl(double, [[.8, .2], [.2, .8]])->log;

}

##-- tests: model 1
sub vtest {
  ($delta,$psi) = hmmviterbi($a,$b,$pi, $o);
  $path = hmmpath($psi, sequence(long,$delta->dim(0)));
}

##-- 1--3: expect for o=[0,1]
testmodel1();
$o = pdl(long,[0,1]);
vtest();

isok("o=[0,1]: delta", all($delta->approx(pdl(double, [[.48,.08],[.048,.1152]])->log)));
isok("o=[0,1]: psi", all($psi == pdl(long, [[0,0],[0,0]])));
isok("o=[0,1]: path", all($path == pdl(long, [[0,0],[0,1]])));

##-- 4--6 expect for o=[0,1,0]
$o=pdl(long,[0,1,0]);
vtest();

isok("o=[0,1,0]: delta", all($delta->approx(pdl(double, [[.48,.08],[.048,.1152],[.0192,.01152]])->log)));
isok("o=[0,1,0]: psi", all($psi == pdl(long, [[0,0],[0,0],[0,1]])));
isok("o=[0,1,0]: path", all($path == pdl(long, [[0,0,0],[0,1,1]])));

print "\n";
# end of t/XX_yyyy.t

