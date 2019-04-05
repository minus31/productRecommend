for blob in $(git fsck --lost-found | awk ‘$2 == “blob” { print $3 }’); do git cat-file -p $blob > $blob.txt; done
