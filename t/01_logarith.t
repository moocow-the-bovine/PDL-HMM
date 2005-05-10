# -*- Mode: Perl -*-
# t/01_logarith.t: test log arithmetic

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

# load common subs
use Test;
do "$TEST_DIR/common.plt";
use PDL;
use PDL::HMM;

BEGIN { plan tests=>11, todo=>[]; }

## 1: test logzero
isok("exp(logzero)==0", exp(logzero)==0);

## 2--3: test log add
sub addlogs { my ($x,$y)=@_; return exp(logadd(pdl(double,$x),pdl(double,$y)))->approx($x+$y); }
isok("addlogs(2,3)==5", addlogs(2,3)==5);
isok("addlogs(0,1)==1", addlogs(0,1)==1);

## 4--5: test logsumover
sub sumlogs { return exp(logsumover(pdl(double,\@_))->log); }
isok("sumlogs(2,3,4)==9", sumlogs(2,3,4)==9);
isok("sumlogs(0..10)==55", sumlogs(0..10)==55);

## 6-8: test log difference
sub sublogs { my ($x,$y)=@_; return exp(logdiff(pdl(double,$x),pdl(double,$y))); }
isok("logdiff(1,0)==1", sublogs(1,0)->approx(1));
isok("logdiff(100,99)==1", sublogs(100,99)->approx(1));
isok("logdiff(1e-10,1e-11)", logdiff(1e-10,1e-11)->approx(9e-11));

##-- 9-11: symmetric difference?
isok("logdiff(0,1)==1", sublogs(0,1)->approx(1));
isok("logdiff(99,100)==1", sublogs(99,100)->approx(1));
isok("logdiff(1e-11,1e-10)", logdiff(1e-11,1e-10)->approx(9e-11));

print "\n";
# end of t/01_ini.t

