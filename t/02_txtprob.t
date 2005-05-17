# -*- Mode: Perl -*-
# t/02_txtprob.t: test text-probability

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

# load common subs
use Test;
do "$TEST_DIR/common.plt";
use PDL;
use PDL::HMM;

BEGIN { plan tests=>4, todo=>[]; }


## test model 1:
sub testmodel1 {
  #$n = 2; $k = 2;

  $a = pdl(double, [[.5,.5],[.5,.5]])->log;
  $b = pdl(double, [[1,0],[0,1]])->log;
  $pi = pdl(double, [.5,.5])->log;
  $omega = pdl(double, [.5,.5])->log;
}

sub testtp {
  my $o=shift;
  $fw = hmmfw($a,$b,$pi, $o);
  $fwtp = logsumover($fw->slice(",-1") + $omega);

  $bw   = hmmbw($a,$b,$omega, $o);
  $bwtp = logsumover($bw->slice(",0") + $pi + $b->slice(",(".$o->at(0).")"));
}


##-- 1--4: model 1
testmodel1;
testtp(pdl([0,1]));

isok("model-1: alpha",      all($fw->approx(pdl(double, [[1/2,0],[0,1/4]])->log)));
isok("model-1: beta",       all($bw->approx(pdl(double, [[1/4,1/4],[1/2,1/2]])->log)));
isok("model-1: txtprob",    all($fwtp->approx(1/8)));
isok("model-1: p_alpha~p_beta", all($bwtp->approx($fwtp)));


print "\n";
# end of t/XX_yyyy.t

